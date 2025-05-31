import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image

def load_model(model_path=r"models\model.keras"):
    return keras.models.load_model(model_path)

print(f"Loading Segmentation Model")
model = load_model()

def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

def segment_image(image_path):
    input_image = preprocess_image(image_path)
    pred_mask = model.predict(input_image)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8).squeeze()
    return pred_mask, input_image[0].squeeze()

def mask_overlay_original(image_path):
    pred_mask, input_image = segment_image(image_path)
    multiply_image = input_image * pred_mask

    # Normalize mask and apply jet colormap
    cmap = plt.get_cmap('jet')
    norm_mask = (pred_mask - np.min(pred_mask)) / (np.max(pred_mask) - np.min(pred_mask) + 1e-8)
    color_mask = cmap(norm_mask)[..., :3]

    # Convert input image to RGB if grayscale
    input_rgb = np.stack([input_image]*3, axis=-1)
    overlay = (0.6 * input_rgb + 0.4 * color_mask)
    overlay = np.clip(overlay, 0, 1)

    return input_image, pred_mask, multiply_image, overlay