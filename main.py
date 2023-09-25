import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Load and preprocess your dataset (real and fake images) and labels
# Replace 'real_images' and 'fake_images' with your data loading code.
real_images = "/kaggle/input/realfake/download.jpeg"
fake_images = "/kaggle/input/realfake/download (1).jpeg"
n_real = len(real_images)
n_fake = len(fake_images)
labels_real = np.zeros(n_real)  # Labels for real images (0)
labels_fake = np.ones(n_fake)   # Labels for fake images (1)

# Combine real and fake images and labels
images = np.concatnate((real_images, fake_images))
labels = np.concatnate((labels_real, labels_fake))

# Split the dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42)

# Define a CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, image_channels)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification (real vs. fake)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# Make predictions on new images
new_images = load_and_preprocess_new_images()  # Replace with your image loading code
predictions = model.predict(new_images)

# Set a threshold to classify as real or fake based on the prediction
threshold = 0.5
labels = ["Real" if prediction < threshold else "Fake" for prediction in predictions]

# Print the classification results
for i, label in enumerate(labels):
    print(f"Image {i + 1}: {label} ({predictions[i][0]:.2f})")
print("Images Shape:", images.shape)
print("Labels Shape:", labels.shape)
