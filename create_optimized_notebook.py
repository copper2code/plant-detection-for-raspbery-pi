import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

cells = []

# ─── MARKDOWN: Title ───────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""# 🌿 Plant Disease Classification — GTX 1650 Optimized
**Hardware:** i5 11th Gen | 16 GB RAM | GTX 1650 Mobile (4 GB VRAM)

Optimizations applied:
- ✅ AMP (Automatic Mixed Precision) — ~35% less VRAM
- ✅ Gradient accumulation — keeps effective batch large without OOM
- ✅ `pin_memory=True` + `non_blocking=True` — faster CPU→GPU transfers
- ✅ `cudnn.benchmark=True` — CUDA kernel auto-tuning
- ✅ `num_workers=4` + `persistent_workers=True` — parallel data loading
- ✅ Gradient clipping — training stability with AMP
- ✅ Image size 224×224 (standard, saves VRAM vs 256×256)
"""))

# ─── CELL 1: Imports + kagglehub download ──────────────────────────────────────
cells.append(new_code_cell("""\
import os, json, pickle, random, warnings
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import kagglehub

warnings.filterwarnings('ignore')

# ── CUDA tuning for GTX 1650 ──────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True

# ── Hardware config ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4          # 16 GB RAM → 4 workers is safe
PIN_MEMORY  = True       # requires num_workers > 0
IMAGE_SIZE  = (224, 224) # standard ImageNet size — saves VRAM vs 256×256
BATCH_SIZE  = 32         # fits in 4 GB VRAM with AMP enabled
GRAD_ACCUM  = 2          # effective batch = 32×2 = 64 without extra VRAM

if DEVICE.type == "cuda":
    props = torch.cuda.get_device_properties(DEVICE)
    print(f"GPU  : {props.name}")
    print(f"VRAM : {props.total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  CUDA not available — running on CPU")

# ── Download dataset via kagglehub ─────────────────────────────────────────────
print("\\n⬇️  Downloading PlantVillage dataset…")
DATASET_ROOT = kagglehub.dataset_download("emmarex/plantdisease")
print(f"Dataset root : {DATASET_ROOT}")

# Locate the PlantVillage folder inside the download
DATA_DIR = None
for root, dirs, files in os.walk(DATASET_ROOT):
    if "PlantVillage" in dirs:
        DATA_DIR = os.path.join(root, "PlantVillage")
        break
    # also handle if we ARE inside the folder already
    if os.path.basename(root) == "PlantVillage":
        DATA_DIR = root
        break

if DATA_DIR is None:
    # fallback: just use the root and hope sub-folders are disease classes
    DATA_DIR = DATASET_ROOT

print(f"Using DATA_DIR: {DATA_DIR}")
print(f"Classes found: {len([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])}")
"""))

# ─── CELL 2: Visualise samples ─────────────────────────────────────────────────
cells.append(new_code_cell("""\
def display_disease_samples(data_dir, plants=None, num_cols=5):
    disease_folders = sorted([
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f))
    ])
    if plants is not None:
        disease_folders = [f for f in disease_folders if any(p in f for p in plants)]

    num_diseases = len(disease_folders)
    num_rows = (num_diseases + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))

    if num_rows == 1:
        axes = [axes] if num_cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    i = 0
    for i, disease_folder in enumerate(disease_folders):
        folder_path = os.path.join(data_dir, disease_folder)
        img_files = [f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if img_files:
            img_path = os.path.join(folder_path, random.choice(img_files))
            img = Image.open(img_path).convert('RGB')
            axes[i].imshow(img)
            axes[i].set_title(disease_folder.replace('_', ' '), fontsize=9)
            axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

print("🌿 Sample images from plant disease categories:")
display_disease_samples(DATA_DIR)
"""))

# ─── CELL 3: Dataset class ─────────────────────────────────────────────────────
cells.append(new_code_cell("""\
class PlantDiseaseDataset(Dataset):
    \"\"\"Custom Dataset for loading plant disease images.\"\"\"
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img   = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)
"""))

# ─── CELL 4: Model ─────────────────────────────────────────────────────────────
cells.append(new_code_cell("""\
try:
    from torchinfo import summary
except ImportError:
    summary = None
    print("torchinfo not installed — skipping model summary (pip install torchinfo)")


class PlantDiseaseModel(nn.Module):
    \"\"\"Custom CNN for plant disease classification.\"\"\"
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding="same"),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2)
            )

        self.conv_block1    = conv_block(3,   64)
        self.conv_block2    = conv_block(64,  128)
        self.conv_block3    = conv_block(128, 256)
        self.conv_block4    = conv_block(256, 512)
        self.conv_block5    = conv_block(512, 512)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.global_avg_pool(x)
        x = self.fc_block(x)
        return x

# Quick sanity-check
if summary is not None:
    print(summary(PlantDiseaseModel(15), input_size=(1, 3, 224, 224)))
else:
    _m = PlantDiseaseModel(15)
    print(f"Model params: {sum(p.numel() for p in _m.parameters()):,}")
    del _m
"""))

# ─── CELL 5: EarlyStopping ─────────────────────────────────────────────────────
cells.append(new_code_cell("""\
class EarlyStopping:
    \"\"\"Stop training when validation loss stops improving.\"\"\"
    def __init__(self, patience=5, min_delta=0.001, save_path="best_model.pth"):
        self.patience  = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter   = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"  [Checkpoint saved → {self.save_path}]")
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True   # trigger stop
        return False
"""))

# ─── CELL 6: Data loading ──────────────────────────────────────────────────────
cells.append(new_code_cell("""\
def load_images(directory_root):
    image_list, label_list = [], []
    print("[INFO] Scanning images…")
    for disease_folder in sorted(os.listdir(directory_root)):
        folder_path = os.path.join(directory_root, disease_folder)
        if not os.path.isdir(folder_path):
            continue
        for img_name in os.listdir(folder_path):
            if img_name.startswith("."):
                continue
            img_path = os.path.join(folder_path, img_name)
            if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_list.append(img_path)
                label_list.append(disease_folder)
    print(f"[INFO] Found {len(image_list)} images across {len(set(label_list))} classes")
    return image_list, label_list


def prepare_data(directory_root, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                 test_size=0.3, valid_ratio=0.5, random_state=42):
    image_paths, labels = load_images(directory_root)

    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    class_names = list(le.classes_)
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)

    # Splits
    tr_paths, tmp_paths, tr_lbl, tmp_lbl = train_test_split(
        image_paths, labels_enc, test_size=test_size,
        random_state=random_state, stratify=labels_enc)
    va_paths, te_paths, va_lbl, te_lbl = train_test_split(
        tmp_paths, tmp_lbl, test_size=valid_ratio,
        random_state=random_state, stratify=tmp_lbl)

    print(f"Train: {len(tr_paths)} | Val: {len(va_paths)} | Test: {len(te_paths)}")

    train_tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    with open('inference_transform.pkl', 'wb') as f:
        pickle.dump(eval_tf, f)

    # ── Optimised DataLoaders for GTX 1650 + 16 GB RAM ───────────────────────
    dl_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0)
    )

    train_loader = DataLoader(
        PlantDiseaseDataset(tr_paths, tr_lbl, train_tf),
        batch_size=batch_size, shuffle=True, **dl_kwargs)
    valid_loader = DataLoader(
        PlantDiseaseDataset(va_paths, va_lbl, eval_tf),
        batch_size=batch_size, shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(
        PlantDiseaseDataset(te_paths, te_lbl, eval_tf),
        batch_size=batch_size, shuffle=False, **dl_kwargs)

    return train_loader, valid_loader, test_loader, len(class_names)
"""))

# ─── CELL 7: Dataset distribution plot ────────────────────────────────────────
cells.append(new_code_cell("""\
def plot_dataset_distribution(data_dir):
    if not os.path.exists(data_dir):
        print(f"❌ Path not found: {data_dir}")
        return

    folders = [f for f in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, f))]
    if not folders:
        print("❌ No class folders found")
        return

    counts = {}
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        counts[folder] = len([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    plant_data = {}
    for folder, count in counts.items():
        plant = folder.split("__")[0] if "__" in folder else folder.split("_")[0]
        plant = plant.replace("_", " ")
        status = "Healthy" if "healthy" in folder.lower() else "Diseased"
        if plant not in plant_data:
            plant_data[plant] = {"Healthy": 0, "Diseased": 0}
        plant_data[plant][status] += count

    plants          = list(plant_data.keys())
    healthy_counts  = [plant_data[p]["Healthy"]  for p in plants]
    diseased_counts = [plant_data[p]["Diseased"] for p in plants]

    plt.figure(figsize=(14, 6))
    plt.bar(plants, healthy_counts,  label="Healthy")
    plt.bar(plants, diseased_counts, bottom=healthy_counts, label="Diseased")
    plt.xticks(rotation=45, ha='right')
    plt.title("Plant Dataset Distribution")
    plt.ylabel("Images")
    plt.legend()
    plt.tight_layout()
    plt.show()

    total = sum(healthy_counts) + sum(diseased_counts)
    print(f"\\nDataset Summary")
    print(f"Total images : {total}")
    print(f"Healthy      : {sum(healthy_counts)}")
    print(f"Diseased     : {sum(diseased_counts)}")
    print(f"Plant types  : {len(plants)}")

# Note: direct call works fine in Jupyter (no __main__ guard needed)
plot_dataset_distribution(DATA_DIR)
"""))

# ─── CELL 8: Augmentation visualisation ───────────────────────────────────────
cells.append(new_code_cell("""\
def show_augmentations(data_dir, num_plants=3):
    disease_folders = [f for f in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, f))]
    selected = random.sample(disease_folders, min(num_plants, len(disease_folders)))

    augmentations = [
        ("Original",         transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])),
        ("H-Flip",           transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor()])),
        ("Rotation 30°",     transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.RandomRotation(30), transforms.ToTensor()])),
        ("Color Jitter",     transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ColorJitter(0.2,0.2,0.2), transforms.ToTensor()])),
        ("Combined",         transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.RandomHorizontalFlip(), transforms.RandomRotation(20), transforms.ColorJitter(0.1,0.1), transforms.ToTensor()])),
    ]

    fig, axes = plt.subplots(len(selected), len(augmentations),
                             figsize=(18, 4 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for i, folder in enumerate(selected):
        folder_path = os.path.join(data_dir, folder)
        img_files   = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not img_files:
            continue
        orig_img = Image.open(os.path.join(folder_path, random.choice(img_files))).convert('RGB')

        for j, (name, tf) in enumerate(augmentations):
            img_np = tf(orig_img).permute(1, 2, 0).numpy()
            ax = axes[i][j]
            ax.imshow(img_np)
            if i == 0:
                ax.set_title(name, fontsize=11)
            if j == 0:
                ax.set_ylabel(folder.replace('_', ' '), fontsize=9)
            ax.axis('off')

    plt.suptitle("Data Augmentation Techniques", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.show()

show_augmentations(DATA_DIR)
"""))

# ─── CELL 9: Train & Evaluate — AMP optimised ─────────────────────────────────
cells.append(new_code_cell("""\
# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        bar = tqdm(enumerate(data_loader), desc="Evaluating",
                   total=len(data_loader), leave=False)
        for _, (inputs, labels) in bar:
            # non_blocking requires pin_memory=True on DataLoader
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            bar.set_postfix({"Loss": f"{loss.item():.4f}",
                             "Acc" : f"{correct/total*100:.2f}%"})

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return val_loss / len(data_loader), correct / total * 100, \\
           np.array(all_preds), np.array(all_labels)


# ── Training loop — AMP + gradient accumulation ───────────────────────────────
def train_model(model, train_loader, valid_loader, criterion, optimizer,
                scheduler=None, epochs=25, early_stopping=None,
                device=DEVICE, grad_accum_steps=GRAD_ACCUM):
    model.to(device)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    train_losses, valid_losses, valid_accuracies = [], [], []

    print(f"Training on : {device}  |  Effective batch: {train_loader.batch_size * grad_accum_steps}\\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        bar = tqdm(enumerate(train_loader),
                   desc=f"Epoch {epoch+1:>3}/{epochs}",
                   total=len(train_loader), leave=False)

        for batch_idx, (inputs, labels) in bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward — AMP context
            with autocast(enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss    = criterion(outputs, labels) / grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or \\
               (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * grad_accum_steps
            bar.set_postfix({"Loss": f"{loss.item()*grad_accum_steps:.4f}"})

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss, val_acc, _, _ = evaluate_model(model, valid_loader, criterion, device)
        valid_losses.append(val_loss)
        valid_accuracies.append(val_acc)

        vram = ""
        if device.type == "cuda":
            vram = f"  VRAM: {torch.cuda.memory_reserved(device)/1024**3:.2f} GB"

        print(f"Epoch {epoch+1:>3}/{epochs}  |  "
              f"Train: {train_loss:.4f}  |  "
              f"Val: {val_loss:.4f}  |  "
              f"Acc: {val_acc:.2f}%{vram}")

        if scheduler:
            scheduler.step(val_loss)

        if early_stopping and early_stopping(val_loss, model):
            print("[INFO] Early stopping triggered.")
            break

    save_learning_curves(train_losses, valid_losses, valid_accuracies)
    return train_losses, valid_losses, valid_accuracies


def save_learning_curves(train_losses, valid_losses, valid_accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Val Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.title('Training & Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(valid_accuracies, color='green', label='Val Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)')
    plt.legend(); plt.title('Validation Accuracy')

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150)
    plt.close()
    print("[INFO] Saved learning_curves.png")
"""))

# ─── CELL 10: Inference + main train() ────────────────────────────────────────
cells.append(new_code_cell("""\
def predict_image(model, image_path, transform, device, label_encoder=None):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        with autocast(enabled=(device.type == "cuda")):
            outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)

    idx        = predicted.item()
    confidence = probs[0][idx].item() * 100
    if label_encoder:
        return label_encoder.inverse_transform([idx])[0], confidence, probs[0].cpu().numpy()
    return idx, confidence, probs[0].cpu().numpy()


def train(data_dir, model_save_path="best_model.pth",
          batch_size=BATCH_SIZE, epochs=30,
          learning_rate=0.001, image_size=IMAGE_SIZE):
    \"\"\"Full training pipeline — optimised for GTX 1650 + 16 GB RAM.\"\"\"

    train_loader, valid_loader, test_loader, num_classes = prepare_data(
        data_dir, image_size=image_size, batch_size=batch_size)

    print(f"Classes      : {num_classes}")
    print(f"Using device : {DEVICE}")

    model     = PlantDiseaseModel(num_classes=num_classes, dropout_rate=0.5).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=7, min_delta=0.001,
                                   save_path=model_save_path)

    train_model(model=model, train_loader=train_loader,
                valid_loader=valid_loader, criterion=criterion,
                optimizer=optimizer, scheduler=scheduler,
                epochs=epochs, early_stopping=early_stopping, device=DEVICE)

    # Load best weights
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))

    print("\\n[INFO] Evaluating on test set…")
    test_loss, test_acc, preds, true_lbl = evaluate_model(
        model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.2f}%")

    # Export to ONNX
    dummy = torch.randn(1, 3, *image_size).to(DEVICE)
    torch.onnx.export(model, dummy, "plant_disease_model.onnx",
                      opset_version=11, verbose=False)

    model_config = dict(
        image_size=image_size, num_classes=num_classes,
        model_path=model_save_path,
        label_encoder_path="label_encoder.pkl",
        transform_path="inference_transform.pkl",
        class_names_path="class_names.json"
    )
    with open("model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    print("[INFO] All files saved — ready for deployment.")
    return model, model_config
"""))

# ─── CELL 11: Run training ─────────────────────────────────────────────────────
cells.append(new_code_cell("""\
# ── Kick off training ─────────────────────────────────────────────────────────
MODEL_PATH     = "best_model.pth"
TRAIN_EPOCHS   = 30        # reduce to 10-15 for a quick test
LEARNING_RATE  = 0.00065

model, model_config = train(
    data_dir       = DATA_DIR,
    model_save_path= MODEL_PATH,
    batch_size     = BATCH_SIZE,
    epochs         = TRAIN_EPOCHS,
    learning_rate  = LEARNING_RATE,
    image_size     = IMAGE_SIZE
)
"""))

# ─── CELL 12: Learning curves ─────────────────────────────────────────────────
cells.append(new_code_cell("""\
def visualize_learning_curves():
    try:
        img = plt.imread('learning_curves.png')
        plt.figure(figsize=(12, 5))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        print("learning_curves.png not found — train the model first.")

print("📊 Training Progress:")
visualize_learning_curves()
"""))

# ─── CELL 13: Per-plant accuracy ──────────────────────────────────────────────
cells.append(new_code_cell("""\
def evaluate_by_plant_type(model, test_loader, label_encoder, device):
    model.eval()
    class_correct, class_total = {}, {}
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Per-plant eval"):
            inputs  = inputs.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            with autocast(enabled=(device.type == "cuda")):
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i, label in enumerate(labels):
                idx  = label.item()
                name = label_encoder.inverse_transform([idx])[0]
                class_correct.setdefault(name, 0)
                class_total.setdefault(name, 0)
                class_total[name]   += 1
                class_correct[name] += int(preds[i] == label)

    # Aggregate by plant species
    plants = {}
    for cls_name, correct in class_correct.items():
        sp = cls_name.split("__")[0].replace("_", " ") if "__" in cls_name else cls_name.split("_")[0]
        plants.setdefault(sp, {"correct": 0, "total": 0})
        plants[sp]["correct"] += correct
        plants[sp]["total"]   += class_total[cls_name]

    plant_acc = {p: s["correct"]/s["total"]*100 for p, s in plants.items()}
    sorted_pa = dict(sorted(plant_acc.items(), key=lambda x: x[1], reverse=True))

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(16, 7))
    colors  = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_pa)))
    bars    = ax.bar(list(sorted_pa.keys()), list(sorted_pa.values()),
                     color=colors, edgecolor='#505050', linewidth=1, width=0.7)

    avg = np.mean(list(plant_acc.values()))
    for bar, acc in zip(bars, sorted_pa.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(avg, color='#e74c3c', linewidth=2, alpha=0.8,
               label=f'Avg: {avg:.1f}%')
    ax.legend(fontsize=12)
    ax.set_title("Model Accuracy by Plant Type", fontsize=16, fontweight='bold')
    ax.set_xlabel("Plant Species", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, max(sorted_pa.values()) * 1.15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    return plant_acc
"""))

# ─── CELL 14: Load model + run per-plant eval ─────────────────────────────────
cells.append(new_code_cell("""\
# Load artefacts saved during training
with open('class_names.json')    as f: class_names    = json.load(f)
with open('label_encoder.pkl', 'rb') as f: label_encoder = pickle.load(f)
with open('inference_transform.pkl', 'rb') as f: inf_transform = pickle.load(f)

num_classes = len(class_names)
loaded_model = PlantDiseaseModel(num_classes=num_classes)
loaded_model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
loaded_model.to(DEVICE)
print(f"✅ Model loaded — {num_classes} classes  |  device: {DEVICE}")

# Re-create test_loader (same split)
_, _, test_loader, _ = prepare_data(
    DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    test_size=0.3, valid_ratio=0.5)

print("\\n📊 Per-plant accuracy:")
plant_accuracy = evaluate_by_plant_type(loaded_model, test_loader, label_encoder, DEVICE)
"""))

# ─── CELL 15: Grad-CAM ────────────────────────────────────────────────────────
cells.append(new_code_cell("""\
def apply_gradcam(model, img_path, transform, label_encoder, device,
                  layer_name='conv_block5'):
    try:
        import cv2
    except ImportError:
        print("Install OpenCV first:  pip install opencv-python")
        return

    model.eval()
    activations, gradients = None, None

    def fwd_hook(module, inp, out):
        nonlocal activations
        activations = out.detach()
    def bwd_hook(module, gi, go):
        nonlocal gradients
        gradients = go[0].detach()

    layer_map = {'conv_block5': model.conv_block5[0],
                 'conv_block4': model.conv_block4[0],
                 'conv_block3': model.conv_block3[0]}
    tgt = layer_map.get(layer_name, model.conv_block5[0])
    fh = tgt.register_forward_hook(fwd_hook)
    bh = tgt.register_backward_hook(bwd_hook)

    try:
        img    = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)

        output    = model(tensor)
        pred_idx  = output.argmax(dim=1).item()
        pred_cls  = label_encoder.inverse_transform([pred_idx])[0]

        model.zero_grad()
        output[:, pred_idx].backward()

        if activations is not None and gradients is not None:
            pooled  = torch.mean(gradients, dim=[0, 2, 3])
            for i in range(activations.size(1)):
                activations[:, i, :, :] *= pooled[i]

            heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            if heatmap.max() > 0:
                heatmap /= heatmap.max()

            orig = np.array(img)
            hmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
            hmap = cv2.applyColorMap(np.uint8(255 * hmap), cv2.COLORMAP_JET)
            over = cv2.addWeighted(orig, 0.6, hmap, 0.4, 0)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(orig);  axes[0].set_title("Original"); axes[0].axis('off')
            axes[1].imshow(cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB))
            axes[1].set_title("Grad-CAM");  axes[1].axis('off')
            axes[2].imshow(cv2.cvtColor(over, cv2.COLOR_BGR2RGB))
            axes[2].set_title(f"Pred: {pred_cls}"); axes[2].axis('off')
            plt.tight_layout(); plt.show()
    finally:
        fh.remove(); bh.remove()


def generate_gradcam_visualizations(num_samples=2):
    print("\\n🔍 Grad-CAM visualisations…")
    samples = []
    for folder in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(fpath):
            continue
        imgs = [f for f in os.listdir(fpath)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if imgs:
            samples.append((os.path.join(fpath, random.choice(imgs)), folder))

    for img_path, label in random.sample(samples, min(num_samples, len(samples))):
        print(f"  → {label}")
        apply_gradcam(loaded_model, img_path, inf_transform, label_encoder, DEVICE)

generate_gradcam_visualizations(num_samples=2)
"""))

# ── Build & write notebook ─────────────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"}
}

output_path = "plant-disease-classifier-gtx1650-optimized.ipynb"
with open(output_path, 'w') as f:
    nbformat.write(nb, f)

print(f"✅ Notebook written to: {output_path}")
