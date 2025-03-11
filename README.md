# CNN-Based MNIST Digit Classifier 
## Overview
This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The model is trained, evaluated, and tested with predictions made on unseen data.
Dataset
•	MNIST Dataset: A dataset of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels.
•	TensorFlow provides built-in access to MNIST using: 
•	from tensorflow.keras.datasets import mnist
•	(x_train, y_train), (x_test, y_test) = mnist.load_data()
## Data Preprocessing
1.	Normalization: Rescaled pixel values from 0-255 to 0-1 to improve model training: 
2.	x_train, x_test = x_train / 255.0, x_test / 255.0
3.	Reshaping: Converted images to the required shape (28, 28, 1) for CNN compatibility: 
4.	x_train = x_train.reshape(-1, 28, 28, 1)
5.	x_test = x_test.reshape(-1, 28, 28, 1)
## Model Architecture
The CNN consists of:
•	Convolutional Layers (Conv2D): Extracts features using filters.

•	MaxPooling Layer (MaxPooling2D): Reduces spatial dimensions.

•	Flatten Layer (Flatten): Converts 2D feature maps into a 1D array.

•	Fully Connected Layers (Dense): Processes and classifies features.

•	Activation Functions: Used ReLU for hidden layers and Softmax for output.
## Model Definition
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
## Model Compilation & Training
•	Loss Function: sparse_categorical_crossentropy for multi-class classification.

•	Optimizer: adam for efficient training.

•	Metrics: Tracked accuracy.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
## Model Evaluation & Visualization
•	Plotted training vs. validation accuracy to check for overfitting.

•	Key Observations:

o	The training accuracy increased steadily.

o	Validation accuracy remained close to training accuracy (no significant overfitting).
## Making Predictions
predictions = model.predict(x_test)
print("Predicted label for first test image:", predictions[0].argmax())
•	The model outputs probability scores for each class (0-9), and we select the highest probability using argmax().
## Conclusion
✅ Successfully trained a CNN model for digit classification.

✅ Achieved high accuracy (~99%) with minimal overfitting.

✅ Used visualization techniques to evaluate model performance.

