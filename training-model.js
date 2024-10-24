const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Constants for image directories, batch size, and image size
const trainDir = "./train";  // Update train directory path
const batchSize = 32;
const imageHeight = 256;
const imageWidth = 256;

// Function to load and preprocess image data
async function loadImages(dir) {
    const images = [];
    const labels = [];

    // Read subdirectories (Formalin and Non formalin)
    const categories = fs.readdirSync(dir);

    for (const category of categories) {
        const categoryPath = path.join(dir, category);

        // Skip if it's not a directory (e.g., .DS_Store)
        if (!fs.statSync(categoryPath).isDirectory()) {
            continue;
        }

        // Determine the label based on folder name (0 = Non formalin, 1 = Formalin)
        const label = category.toLowerCase() === 'formalin' ? 1 : 0;

        // Read image files from the subdirectory
        const files = fs.readdirSync(categoryPath);

        for (const file of files) {
            const filePath = path.join(categoryPath, file);

            // Skip non-image files (like .DS_Store)
            if (!file.toLowerCase().endsWith('.jpg') && !file.toLowerCase().endsWith('.png')) {
                continue;
            }

            const imageBuffer = fs.readFileSync(filePath);
            let imageTensor = tf.node.decodeImage(imageBuffer);

            // Convert RGBA (4 channels) to RGB (3 channels) if necessary
            if (imageTensor.shape[2] === 4) {
                imageTensor = imageTensor.slice([0, 0, 0], [-1, -1, 3]);  // Remove the alpha channel
            }

            imageTensor = imageTensor
                .resizeNearestNeighbor([imageHeight, imageWidth])
                .toFloat()
                .div(tf.scalar(255.0));

            images.push(imageTensor);
            labels.push(label);
        }
    }

    // Stack tensors into batches
    const imageBatch = tf.stack(images);
    const labelBatch = tf.tensor1d(labels, 'int32');
    return { images: imageBatch, labels: labelBatch };
}




// Build the model
function createModel() {
    const model = tf.sequential();

    // Add layers (equivalent to Keras code)
    model.add(tf.layers.rescaling({ scale: 1. / 255, inputShape: [imageHeight, imageWidth, 3] }));
    model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

    model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

    model.add(tf.layers.conv2d({ filters: 256, kernelSize: 3, activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

    model.add(tf.layers.flatten());

    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));  // Binary classification

    // Compile model
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
    });

    return model;
}

// Train the model
async function trainModel() {
    const { images: trainImages, labels: trainLabels } = await loadImages(trainDir);

    const model = createModel();

    // Train the model
    await model.fit(trainImages, trainLabels, {
        epochs: 20,
        batchSize,
        validationSplit: 0.2,
    });

    // Save the model to a directory
    await model.save('file://./model');
    
    console.log('Model trained and saved.');
}

trainModel().catch(console.error);