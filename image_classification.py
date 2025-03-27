import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt  # Import Matplotlib
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the image
image_path = "sample.jpg"
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
img_resized = cv2.resize(img, (224, 224))  # Resize to 224x224
img_preprocessed = preprocess_input(img_resized)  # Normalize pixel values
img_preprocessed = np.expand_dims(img_preprocessed, axis=0)  # Add batch dimension

# Display the image using Matplotlib
plt.imshow(img)  # Show the original image
plt.axis("off")  # Hide axis
plt.title("Input Image")  # Add a title
plt.show()  # Show the figure

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Make predictions
predictions = model.predict(img_preprocessed)

# Decode predictions
decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)

# Print the top 3 predictions
print("\nTop Predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
    print(f"{i+1}. {label}: {score:.4f}")
