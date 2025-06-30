import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import os

# --- Konfigurasi Aplikasi Streamlit ---
st.set_page_config(
    page_title="Website Klasifikasi Sampah",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("♻️ Klasifikasi Jenis Sampah")
st.write("Website ini mengklasifikasikan jenis sampah ke dalam kategori yang telah dilatih.")
st.write("---")

# --- Path Model dan Class Indices ---
MODEL_PATH = 'mobilenetv2_sampah.h5'
CLASS_INDICES_PATH = 'class_indices.json'
TARGET_SIZE = (224, 224) 

@st.cache_resource
def load_my_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

@st.cache_resource
def load_class_indices():
    try:
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
            idx_to_class = {str(v): k for k, v in class_indices.items()}
            return idx_to_class
    except Exception as e:
        st.error(f"Gagal memuat class_indices.json: {e}")
        st.stop()

model = load_my_model()
idx_to_class = load_class_indices()

# --- Fungsi Preprocessing Gambar ---
def preprocess_image(image):
    image = image.resize(TARGET_SIZE)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0
    return image_array

# --- Unggah Gambar oleh Pengguna atau Ambil Langsung ---
st.subheader("Pilih Sumber Gambar:")

# Opsi 1: Unggah dari File
uploaded_file = st.file_uploader("Unggah gambar dari perangkat Anda", type=["jpg", "jpeg", "png"])

# Opsi 2: Ambil Gambar Langsung dari Kamera
# Kita bisa menempatkannya di bawah file_uploader
st.markdown("---") # Untuk pemisah visual
camera_image = st.camera_input("Ambil gambar langsung dari kamera")

# Variabel untuk menyimpan gambar input yang akan diproses
input_image = None
image_caption = ""

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    image_caption = 'Gambar yang Diunggah'
elif camera_image is not None:
    input_image = Image.open(camera_image)
    image_caption = 'Gambar dari Kamera'

# Logika Prediksi hanya akan berjalan jika ada input_image
if input_image is not None:
    st.image(input_image, caption=image_caption, use_container_width=True)
    st.write("")
    st.write("Memprediksi...")

    processed_image = preprocess_image(input_image)

    try:
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class_name = idx_to_class.get(str(predicted_class_idx), "Unknown Class")
        
        st.success(f"Gambar ini diklasifikasikan sebagai: **{predicted_class_name.upper()}**")

        st.subheader("Detail Probabilitas:")
        probabilities_percent = np.round(predictions[0] * 100, 2)
        sorted_indices = np.argsort(probabilities_percent)[::-1]
        
        display_data = []
        for i in sorted_indices:
            class_name = idx_to_class.get(str(i), f"Class {i}")
            probability = probabilities_percent[i]
            display_data.append({"Kategori Sampah": class_name, "Probabilitas (%)": f"{probability:.2f}"})
        
        st.dataframe(display_data)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")

else: # Ini akan tampil jika tidak ada gambar yang diunggah atau diambil
    st.info("Mohon unggah gambar atau ambil gambar dari kamera untuk memulai klasifikasi.")

st.write("---")
st.markdown("Website dibuat untuk klasifikasi jenis sampah.")
st.write("---")
st.markdown("Jika Anda suka dengan website ini atau ingin memberikan saran dan masukan silahkan kirim ke nelpisaragih2306@gmail.com.")
