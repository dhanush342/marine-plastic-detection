document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const resultContainer = document.getElementById('result-container');
    const videoElement = document.getElementById('video');
    const startButton = document.getElementById('start-button');
    const stopButton = document.getElementById('stop-button');

    // Handle file upload
    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault();
        const formData = new FormData(uploadForm);
        
        fetch('/api/inference', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    // Display results from the model
    function displayResults(data) {
        resultContainer.innerHTML = '';
        if (data.results) {
            data.results.forEach(result => {
                const resultItem = document.createElement('div');
                resultItem.textContent = `Detected: ${result.label} with confidence ${result.confidence}`;
                resultContainer.appendChild(resultItem);
            });
        } else {
            resultContainer.textContent = 'No results found.';
        }
    }

    // Start real-time detection
    startButton.addEventListener('click', function() {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                videoElement.srcObject = stream;
                videoElement.play();
                startDetection(stream);
            })
            .catch(error => {
                console.error('Error accessing webcam:', error);
            });
    });

    // Stop real-time detection
    stopButton.addEventListener('click', function() {
        const stream = videoElement.srcObject;
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        videoElement.srcObject = null;
    });

    // Function to handle real-time detection
    function startDetection(stream) {
        const videoTrack = stream.getVideoTracks()[0];
        const imageCapture = new ImageCapture(videoTrack);

        setInterval(() => {
            imageCapture.grabFrame()
                .then(imageBitmap => {
                    // Convert imageBitmap to blob for sending to the server
                    const canvas = document.createElement('canvas');
                    canvas.width = imageBitmap.width;
                    canvas.height = imageBitmap.height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(imageBitmap, 0, 0);
                    canvas.toBlob(blob => {
                        const formData = new FormData();
                        formData.append('image', blob, 'frame.jpg');

                        fetch('/api/realtime-detection', {
                            method: 'POST',
                            body: formData
                        })
                        .then(response => response.json())
                        .then(data => {
                            displayResults(data);
                        })
                        .catch(error => {
                            console.error('Error:', error);
                        });
                    }, 'image/jpeg');
                })
                .catch(error => {
                    console.error('Error grabbing frame:', error);
                });
        }, 1000); // Adjust the interval as needed
    }
});