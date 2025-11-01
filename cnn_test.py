import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Ganti ini dengan nama folder yang ingin diuji
folder_name = "dataset/test"

# Cari file gambar pertama di folder
def get_first_image_path(folder):
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            return os.path.join(folder, file)
    return None

# Load gambar uji
img_path = get_first_image_path(folder_name)  # ganti dengan path gambar kamu
if img_path is None:
    print("Tidak ada gambar ditemukan di folder:", folder_name)
else:
    # Load model
    model = tf.keras.models.load_model("model_hama_dan_nonhama_mobilenetv2_final.keras")
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalisasi

    # Prediksi
    pred = model.predict(img_array)
    predicted_class = np.argmax(pred)

    pred = model.predict(img_array)
    confidence = np.max(pred)
    predicted_class = np.argmax(pred)

    # Tampilkan hasil
    class_names = ['tomato_healthy', 'tomato_leaf_curl']  # sesuaikan dengan kelas kamu
    # print(f"Prediksi: {class_names[predicted_class]}")

    if confidence < 0.7:
        print("Gambar tidak dikenali dengan cukup yakin.")
    else:
        print(f"Prediksi: {class_names[predicted_class]} ({confidence:.2%})")



