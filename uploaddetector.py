import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from pathlib import Path
from keras.utils import img_to_array, load_img

MODEL_PATH = "mudra_model.keras"
CLASS_NAMES = ["Alapadmam", "Chandrakala", "Kapitham", "Shikaram"]
IMG_SIZE = (224, 224)


model = tf.keras.models.load_model(MODEL_PATH)


def prepare_input(img_path):
    
    img = load_img(img_path, target_size=IMG_SIZE)
    arr = img_to_array(img) 
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_mudra(img_path):
    batch = prepare_input(img_path)
    pred = model.predict(batch, verbose=0)[0]

    top_idx = int(np.argmax(pred))
    mudra = CLASS_NAMES[top_idx]
    conf = float(pred[top_idx] * 100.0)

    print("Class probabilities:")
    for name, prob in zip(CLASS_NAMES, pred):
        print(f"  {name}: {prob * 100:.1f}%")

    return mudra, conf


print("UPLOAD Mudra Detector")
print("Alapadmam | Chandrakala | Kapitham | Shikaram")
print("Pipeline: Raw RGB input (model handles preprocessing internally)")

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Upload Mudra JPG",
    initialdir=r"C:\Users\alfee\handgesture\val",
    filetypes=[("JPG", "*.jpg *.jpeg")],
)
root.destroy()

if file_path:
    
    val_folder = Path(r"C:\Users\alfee\handgesture\val").resolve()
    selected_file = Path(file_path).resolve()
    
    try:
        
        selected_file.relative_to(val_folder)
        
        mudra, conf = predict_mudra(str(selected_file))
        print(f"Prediction: {mudra.upper()} ({conf:.1f}%)")
        if conf < 60:
            print("Warning: low confidence. Retrain with more diverse images or check class labels.")

        img_bgr = cv2.imread(str(selected_file))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 6))
        plt.imshow(img_rgb)
        plt.title(f"{mudra} ({conf:.1f}%)", fontsize=16)
        plt.axis("off")
        plt.show()
    except ValueError:
        
        messagebox.showerror("Invalid File", f"Error: File must be from {val_folder}\n\nSelected: {selected_file}")
        sys.exit(1)
else:
    print("No file selected.")
