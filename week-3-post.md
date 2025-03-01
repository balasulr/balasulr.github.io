---
layout: post
author: Lakshmi Balasubramaniam
tags: [cscc]
---

### [Machine Learning Model]

**Describe how Teachable Machine uses machine learning to come up with solutions to an arbitrary problem:** [Source](https://www.geeksforgeeks.org/machine-learning-model-with-teachable-machine/)
- According to a GeeksforGeeks article, Teachable Machine uses machine learning to come up with solutions to an arbitrary problem by allowing “users to train their own machine learning models without any coding experience” and it uses “a web camera to gather images or videos”. Those images are then used to train a machine learning model. The article also says that the “user can then use the model to classify new images or videos.”

**What application did you choose to create using Teachable Machine?**
- The application I developed using Teachable Machine is designed to differentiate between various types of good and bad fruits and vegetables. The model was trained using examples such as a good banana, a bad banana, a bad peach, a purple onion, an orange and a plum

**What problem is it intended to solve?**
- The problem addresses the challenge of distinguishing between fruits and vegetables that appear similar in size and appearance. Also, determining if they have gone bad. The goal was to test whether the model could accurately differentiate between them

**Why did you choose to create this application?**
- I chose to create this application because I was curious about if the model could differentiate between fruits and vegetables of similar size, shape and if they were going to go bad

**Provide the URL to your application or exported model created using Teachable Machine:**
**URL**: [https://teachablemachine.withgoogle.com/models/nr9naGuub/](https://teachablemachine.withgoogle.com/models/nr9naGuub/)

**Tensorflow.js JavaScript Code snippet:**
<div>Teachable Machine Image Model</div>
<button type="button" onclick="init()">Start</button>
<div id="webcam-container"></div>
<div id="label-container"></div>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@latest/dist/teachablemachine-image.min.js"></script>
<script type="text/javascript">
    // More API functions here:
    // https://github.com/googlecreativelab/teachablemachine-community/tree/master/libraries/image

    // the link to your model provided by Teachable Machine export panel
    const URL = "./my_model/";

    let model, webcam, labelContainer, maxPredictions;

    // Load the image model and setup the webcam
    async function init() {
        const modelURL = URL + "model.json";
        const metadataURL = URL + "metadata.json";

        // load the model and metadata
        // Refer to tmImage.loadFromFiles() in the API to support files from a file picker
        // or files from your local hard drive
        // Note: the pose library adds "tmImage" object to your window (window.tmImage)
        model = await tmImage.load(modelURL, metadataURL);
        maxPredictions = model.getTotalClasses();

        // Convenience function to setup a webcam
        const flip = true; // whether to flip the webcam
        webcam = new tmImage.Webcam(200, 200, flip); // width, height, flip
        await webcam.setup(); // request access to the webcam
        await webcam.play();
        window.requestAnimationFrame(loop);

        // append elements to the DOM
        document.getElementById("webcam-container").appendChild(webcam.canvas);
        labelContainer = document.getElementById("label-container");
        for (let i = 0; i < maxPredictions; i++) { // and class labels
            labelContainer.appendChild(document.createElement("div"));
        }
    }

    async function loop() {
        webcam.update(); // update the webcam frame
        await predict();
        window.requestAnimationFrame(loop);
    }

    // run the webcam image through the image model
    async function predict() {
        // predict can take in an image, video or canvas html element
        const prediction = await model.predict(webcam.canvas);
        for (let i = 0; i < maxPredictions; i++) {
            const classPrediction =
                prediction[i].className + ": " + prediction[i].probability.toFixed(2);
            labelContainer.childNodes[i].innerHTML = classPrediction;
        }
    }
</script>
