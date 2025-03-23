import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Input
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
def load_data(data_dir, mask_dir, img_size=(256, 256)):
    images = []
    masks = []

    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name.replace(".jpg", "_mask.jpg"))  # Adjust mask naming
        
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)  # Load mask in grayscale
        
        img = cv2.resize(img, img_size) / 255.0  # Normalize
        mask = cv2.resize(mask, img_size) / 255.0  # Normalize to [0,1]
        
        images.append(img)
        masks.append(mask.reshape(img_size[0], img_size[1], 1))  # Add channel dim
    
    return np.array(images), np.array(masks)

# Load dataset
IMG_DIR = "F:/CodeClash2025/road_img.jpg"
MASK_DIR = "F:/CodeClash2025/lane_image.jpg"
X, Y = load_data(IMG_DIR, MASK_DIR)

# Split dataset
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Dataset Loaded: {X_train.shape}, {Y_train.shape}")

# U-Net Model
def unet_model(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)

    # Encoder (Downsampling)
    c1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    c1 = Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    c2 = Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
    c3 = Conv2D(256, (3, 3), activation="relu", padding="same")(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(512, (3, 3), activation="relu", padding="same")(p3)
    c4 = Conv2D(512, (3, 3), activation="relu", padding="same")(c4)

    # Decoder (Upsampling)
    u5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c4)
    u5 = Concatenate()([u5, c3])
    c5 = Conv2D(256, (3, 3), activation="relu", padding="same")(u5)
    c5 = Conv2D(256, (3, 3), activation="relu", padding="same")(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = Concatenate()([u6, c2])
    c6 = Conv2D(128, (3, 3), activation="relu", padding="same")(u6)
    c6 = Conv2D(128, (3, 3), activation="relu", padding="same")(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = Concatenate()([u7, c1])
    c7 = Conv2D(64, (3, 3), activation="relu", padding="same")(u7)
    c7 = Conv2D(64, (3, 3), activation="relu", padding="same")(c7)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c7)

    model = keras.Model(inputs, outputs)
    return model

# Compile Model
model = unet_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Train Model
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=20, batch_size=8)

# Save Model
model.save("lane_detection_unet.h5")
print("✅ Model saved successfully!")

# Plot Training History
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.show()
