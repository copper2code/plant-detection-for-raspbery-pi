document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const tabs = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Camera Elements
    const video = document.getElementById('camera-stream');
    const captureBtn = document.getElementById('capture-btn');
    const scanningAnim = document.getElementById('scanning-anim');
    
    // Upload Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadPreview = document.getElementById('upload-preview');
    const analyzeBtn = document.getElementById('analyze-upload-btn');
    const uploadContent = document.querySelector('.upload-content');
    
    // Results Elements
    const resultsEmpty = document.getElementById('results-empty');
    const resultsData = document.getElementById('results-data');
    const resetBtn = document.getElementById('reset-btn');
    
    // State
    let fallbackImageData = null;

    // --- Tab Switching ---
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            tab.classList.add('active');
            const target = document.getElementById(tab.dataset.target);
            target.classList.add('active');
        });
    });

    captureBtn.addEventListener('click', async () => {
        // Visual feedback
        captureBtn.innerHTML = '<i class="ri-loader-4-line ri-spin"></i> Analyzing...';
        captureBtn.disabled = true;
        scanningAnim.classList.remove('hidden');

        try {
            const response = await fetch('/predict_camera', { method: 'POST' });
            const data = await response.json();
            if (response.ok) {
                showResults(data);
            } else {
                alert("Error from server: " + (data.error || "Unknown"));
                resetUI();
            }
        } catch (error) {
            console.error("Network error:", error);
            alert("Failed to connect to the prediction server.");
            resetUI();
        }
    });

    // --- Upload Logic ---
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            fallbackImageData = e.target.result;
            uploadPreview.src = fallbackImageData;
            uploadPreview.classList.remove('hidden');
            uploadContent.classList.add('hidden');
            analyzeBtn.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    analyzeBtn.addEventListener('click', () => {
        if (fallbackImageData) {
            analyzeBtn.innerHTML = '<i class="ri-loader-4-line ri-spin"></i> Analyzing...';
            analyzeBtn.disabled = true;
            analyzeImage(fallbackImageData);
        }
    });

    // --- API & UI Updates ---
    async function analyzeImage(base64Data) {
        try {
            // Hardcoded endpoint due to local development
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Data })
            });

            const data = await response.json();
            
            if (response.ok) {
                showResults(data);
            } else {
                alert("Error from server: " + (data.error || "Unknown"));
                resetUI();
            }
        } catch (error) {
            console.error("Network error:", error);
            alert("Failed to connect to the prediction server.");
            resetUI();
        }
    }

    function showResults(data) {
        // Toggle view
        resultsEmpty.classList.add('hidden');
        resultsData.classList.remove('hidden');
        
        // Populate Title & Confidence
        document.getElementById('res-title').textContent = data.display_name;
        document.getElementById('res-confidence-text').textContent = `${data.confidence}%`;
        
        // Animate confidence bar
        setTimeout(() => {
            document.getElementById('res-confidence-fill').style.width = `${data.confidence}%`;
        }, 100);

        // Populate Status Badge
        const badge = document.getElementById('res-status-badge');
        const iconDict = {
            'healthy': 'ri-checkbox-circle-fill',
            'diseased': 'ri-error-warning-fill',
            'unknown': 'ri-question-fill'
        };
        const classDict = {
            'healthy': 'status-healthy',
            'diseased': data.severity === 'severe' ? 'status-diseased' : 'status-moderate',
            'none': 'status-healthy'
        };
        
        badge.className = 'status-badge ' + (classDict[data.status] || classDict[data.severity] || 'status-unknown');
        badge.innerHTML = `<i class="${iconDict[data.status] || 'ri-information-fill'}"></i> <span>${data.status.toUpperCase()}</span>`;

        // Populate Treatments
        const stepsUl = document.getElementById('res-treatment-steps');
        stepsUl.innerHTML = '';
        data.steps.forEach(step => {
            const li = document.createElement('li');
            li.textContent = step;
            stepsUl.appendChild(li);
        });

        // Prevention
        document.getElementById('res-prevention-text').textContent = data.prevention;

        // Stop scanning visuals
        resetBtnStates();
    }

    function resetUI() {
        resetBtnStates();
        
        // Also clear confidence bar immediately
        document.getElementById('res-confidence-fill').style.width = '0%';
    }

    function resetBtnStates() {
        captureBtn.innerHTML = '<i class="ri-camera-lens-fill"></i> Capture Photo';
        captureBtn.disabled = false;
        scanningAnim.classList.add('hidden');

        analyzeBtn.innerHTML = 'Analyze Photo';
        analyzeBtn.disabled = false;
    }

    resetBtn.addEventListener('click', () => {
        resultsEmpty.classList.remove('hidden');
        resultsData.classList.add('hidden');
        document.getElementById('res-confidence-fill').style.width = '0%';
        
        // Clear upload if in upload tab
        if (!uploadContent.classList.contains('hidden') === false) {
             uploadPreview.classList.add('hidden');
             uploadContent.classList.remove('hidden');
             analyzeBtn.classList.add('hidden');
             fallbackImageData = null;
             fileInput.value = '';
        }
    });

});
