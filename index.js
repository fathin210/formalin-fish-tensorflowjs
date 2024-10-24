const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const cors = require('cors')

const app = express();
const port = 4000;

app.use(cors())
app.use(express.json({
    limit: '10mb'
}));
app.use(express.urlencoded({ extended: true }));

// Load the TensorFlow.js model (not TFLite)
let model;

async function loadModel() {
    const modelPath = path.resolve(__dirname, './model/model.json');
    model = await tf.loadLayersModel(`file://${modelPath}`);
    console.log('Model loaded.');
}

loadModel().catch(console.error);

// Function to preprocess image for the model
async function preprocessImage(imageBuffer) {
    let imageTensor = tf.node.decodeImage(imageBuffer);

    // Convert RGBA (4 channels) to RGB (3 channels) if necessary
    if (imageTensor.shape[2] === 4) {
        imageTensor = imageTensor.slice([0, 0, 0], [-1, -1, 3]);  // Remove the alpha channel
    }

    // Resize image to the expected input size
    imageTensor = imageTensor
        .resizeNearestNeighbor([224, 224])  // Ganti dengan ukuran input yang sesuai
        .toFloat()
        .div(tf.scalar(255.0));

    return imageTensor.expandDims(0);  // Menambahkan dimensi batch
}

// Endpoint untuk inferensi
app.post('/predict', async (req, res) => {
    const { image } = req.body; // Pastikan Anda mengirimkan gambar dalam format yang benar (base64 atau binary)

    try {
        const imageBuffer = Buffer.from(image, 'base64'); // Ubah base64 ke Buffer
        const processedImage = await preprocessImage(imageBuffer);

        const prediction = model.predict(processedImage);
        const probabilities = prediction.dataSync(); // Mendapatkan probabilitas untuk masing-masing kelas

        // Menghitung persentase
        const formalinPercentage = probabilities[0] * 100;
        const nonFormalinPercentage = probabilities[1] * 100;

        // Mengambil hasil prediksi berdasarkan threshold
        const predictedClass = probabilities[0] > 0.5 ? 'formalin' : 'non formalin'; // Ganti threshold sesuai kebutuhan

        // Mengembalikan hasil prediksi
        res.json({
            detail_prediction: {
                formalin: `${formalinPercentage.toFixed(2)}%`,
                non_formalin: `${nonFormalinPercentage.toFixed(2)}%`,
            },
            prediction: predictedClass
        });
    } catch (error) {
        console.error('Error during prediction:', error);
        res.status(500).send('Error during prediction');
    }
});

app.get("/", (req, res) => res.send("Express on Vercel"));

// Jalankan server
app.listen(port, () => {
    console.log(`Server is running at http://localhost:${port}`);
});
