const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const cors = require('cors')

const app = express();
const port = 4000;

app.use(cors())
app.use(express.json({
    limit: '10mb'
}));
app.use(express.urlencoded({ extended: true }));

let model;

async function loadModel() {
    const modelPath = path.resolve(__dirname, './model/model.json');
    model = await tf.loadLayersModel(`file://${modelPath}`);
    console.log('Model loaded.');
}

loadModel().catch(console.error);

async function preprocessImage(imageBuffer) {
    return tf.tidy(() => {
        let imageTensor = tf.node.decodeImage(imageBuffer);

        if (imageTensor.shape[2] === 4) {
            imageTensor = imageTensor.slice([0, 0, 0], [-1, -1, 3]);
        }

        imageTensor = imageTensor
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(tf.scalar(255.0));

        return imageTensor.expandDims(0);
    });
}

app.post('/predict', async (req, res) => {
    const { image } = req.body;
    let processedImage;

    try {
        const imageBuffer = Buffer.from(image, 'base64');
        processedImage = await preprocessImage(imageBuffer);

        const prediction = tf.tidy(() => {
            const output = model.predict(processedImage);
            return output.dataSync();
        });

        const formalinPercentage = prediction[0] * 100;
        const nonFormalinPercentage = prediction[1] * 100;

        const predictedClass = prediction[0] > 0.5 ? 'formalin' : 'non formalin';

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
    } finally {
        tf.dispose([processedImage]);
    }
});

app.get("/", (req, res) => res.send("Express on Vercel"));

app.listen(port, () => {
    console.log(`Server is running at http://localhost:${port}`);
});
