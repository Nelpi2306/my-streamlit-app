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

# --- INJEKSI CSS KUSTOM UNTUK DESAIN ---
st.markdown("""
<style>
/* Kontainer Utama Aplikasi */
.block-container {
    background-color: #f0f8f0; /* Warna hijau sangat tipis (dasar) */
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    border: 1px solid #e0e0e0;
    margin-top: 2rem;
}

/* KOTAK JUDUL BARU */
.title-box {
    background-color: #e0f2e9; /* Warna hijau sedikit lebih pekat */
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 2rem; /* Jarak ke elemen di bawahnya */
    border: 1px solid #c8e6c9;
}

/* Menata teks di dalam Kotak Judul */
.title-box h1 {
    font-size: 2.5rem; /* Ukuran font judul */
    color: #2E7D32;    /* Warna hijau tua untuk teks judul */
    text-align: center;
    margin: 0;
    padding: 0;
}

.title-box p {
    font-size: 1rem;
    color: #455a64;
    text-align: center;
    margin-top: 0.5rem; /* Jarak dari judul ke sub-judul */
}

</style>
""", unsafe_allow_html=True)

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

# Memuat model dan class indices
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

# --- KONTEN APLIKASI ---

# Mengganti st.title dengan blok HTML kustom
title_html = """
<div class="title-box">
    <h1>♻️ Klasifikasi Jenis Sampah</h1>
    <p>Website ini mengklasifikasikan jenis sampah ke dalam kategori yang telah dilatih.</p>
</div>
"""
st.markdown(title_html, unsafe_allow_html=True)

st.divider()
st.subheader("Pilih Sumber Gambar")

uploaded_file = st.file_uploader("Unggah gambar dari perangkat Anda", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("Ambil gambar langsung dari kamera")

input_image = None
image_caption = ""

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    image_caption = 'Gambar yang Diunggah'
elif camera_image is not None:
    input_image = Image.open(camera_image)
    image_caption = 'Gambar dari Kamera'

if input_image is not None:
    st.image(input_image, caption=image_caption, use_container_width=True)
    st.write("")
    
    with st.spinner("Menganalisis gambar..."):
        try:
            processed_image = preprocess_image(input_image)
            predictions = model.predict(processed_image)
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            predicted_class_name = idx_to_class.get(str(predicted_class_idx), "Kelas Tidak Dikenali")
            
            st.success(f"**Hasil Klasifikasi:** **{predicted_class_name.upper()}**")

            st.subheader("Detail Probabilitas:")
            probabilities_percent = np.round(predictions[0] * 100, 2)
            sorted_indices = np.argsort(probabilities_percent)[::-1]
            
            display_data = []
            for i in sorted_indices:
                class_name = idx_to_class.get(str(i), f"Class {i}")
                probability = probabilities_percent[i]
                display_data.append({"Kategori Sampah": class_name, "Probabilitas (%)": f"{probability:.2f}"})
            
            st.dataframe(display_data, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memprediksi: {e}")

else:
    st.info("Silakan unggah gambar atau ambil gambar dari kamera untuk memulai.")

st.divider()
st.markdown("Website ini dibuat untuk klasifikasi jenis sampah.")
st.markdown("Saran dan masukan dapat dikirim ke `nelpisaragih2306@gmail.com`.")
