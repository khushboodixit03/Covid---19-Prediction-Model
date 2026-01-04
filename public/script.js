const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startCamBtn = document.getElementById('startCamBtn');
const captureBtn = document.getElementById('captureBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const photoStatus = document.getElementById('photoStatus');
const resultModal = document.getElementById('resultModal');
const closeModal = document.querySelector('.close');

let stream = null;
let photoData = null;

// Camera access
startCamBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        captureBtn.disabled = false;
        startCamBtn.style.display = 'none';
    } catch (err) {
        alert("Camera access denied or error: " + err);
    }
});

captureBtn.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    photoData = canvas.toDataURL('image/jpeg');

    // Stop camera to save resources
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    video.style.opacity = '0.5';
    photoStatus.classList.remove('hidden');
    captureBtn.disabled = true;
});

analyzeBtn.addEventListener('click', async () => {
    // Collect data
    const patientName = document.getElementById('patientName').value;
    const patientId = document.getElementById('patientId').value;
    const oxygen = document.getElementById('oxygen').value;
    const pulse = document.getElementById('pulse').value;
    const temp = document.getElementById('temperature').value;

    if (!oxygen || !pulse || !temp) {
        alert("Please enter all vital signs.");
        return;
    }

    // Show loading state
    analyzeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
    analyzeBtn.disabled = true;

    const payload = {
        patientName,
        patientId,
        oxygen,
        pulse,
        temperature: temp,
        photo: photoData
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (response.ok) {
            showResult(data);
        } else {
            alert("Error: " + data.error);
        }
    } catch (err) {
        alert("Failed to connect to server.");
        console.error(err);
    } finally {
        analyzeBtn.innerHTML = 'RUN DIAGNOSIS <i class="fa-solid fa-arrow-right"></i>';
        analyzeBtn.disabled = false;
    }
});

function showResult(data) {
    document.getElementById('predictionText').innerText = data.prediction;
    document.getElementById('confidenceText').innerText = data.confidence || "N/A";

    // Setup PDF download
    const pdfLink = document.getElementById('downloadPdf');
    pdfLink.href = "data:application/pdf;base64," + data.pdf_base64;

    resultModal.classList.remove('hidden');
}

closeModal.addEventListener('click', () => {
    resultModal.classList.add('hidden');
});

// Close modal on click outside
window.onclick = function (event) {
    if (event.target == resultModal) {
        resultModal.classList.add('hidden');
    }
}
