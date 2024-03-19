Image Classification with Transfer Learning using MobileNetV2
This repository contains code for image classification using transfer learning with TensorFlow and the MobileNetV2 pre-trained model. Transfer learning is a technique where a model trained on a large dataset (such as ImageNet) is fine-tuned on a specific task, in this case, image classification for a custom dataset.

Description:
Reading and Preprocessing Images: The code utilizes OpenCV to read and preprocess images. It loads images from a specified directory, checks their validity, and visualizes them using Matplotlib.

Data Preparation: Images are organized into classes within directories. The code iterates through each class directory, reads images, resizes them, and prepares the data for training by appending them to a list along with their corresponding labels.

Shuffling and Splitting Data: To ensure randomness, the data is shuffled before splitting into features and labels. Features represent image data, while labels denote the class labels.

Normalization: Image pixel values are normalized to a range between 0 and 1 by dividing by 255.0.

Transfer Learning Model: The MobileNetV2 pre-trained model, available in TensorFlow, is utilized. New layers are added on top of the pre-trained model for fine-tuning, including Dense layers and an output layer with softmax activation for multi-class classification.

Model Compilation and Training: The model is compiled with categorical cross-entropy loss and the Adam optimizer. Training is conducted on the prepared data for a specified number of epochs.

Instructions:
Dataset Preparation: Organize your dataset into class directories, with each class containing its respective images.

Code Execution: Execute the provided Python script in an environment with TensorFlow, OpenCV, and Matplotlib installed. Adjust parameters such as image size, batch size, and number of epochs as needed.

Training and Evaluation: Monitor training progress and model performance through metrics such as accuracy and loss. Evaluate the trained model on validation or test data to assess its generalization capability.

Dependencies:
TensorFlow
OpenCV
Matplotlib
