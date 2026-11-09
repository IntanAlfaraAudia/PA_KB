import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json

# Konfigurasi halaman web
st.set_page_config(
    page_title="Klasifikasi Citra Berry CNN", 
    page_icon="🍓", 
    layout="centered"
)

# Memuat model dan label kelas secara caching agar lebih cepat
@st.cache_resource
def load_model_and_labels():
    model = keras.models.load_model("berry_cnn.keras")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

with st.spinner("Memuat model AI, silakan tunggu sebentar..."):
    model, class_names = load_model_and_labels()

# Antarmuka Pengguna (UI)
st.title("🍓 Aplikasi Klasifikasi Citra Berry")
st.write("Unggah gambar buah berry untuk mendeteksi jenisnya secara otomatis menggunakan model Convolutional Neural Network (CNN) Anda.")

uploaded_file = st.file_uploader("Pilih gambar buah berry (format JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
    
    # Pra-pemrosesan gambar (Preprocessing)
    # Sesuaikan ukuran (150x150) dengan ukuran input saat training model Anda
    img_height, img_width = 150, 150 
    image_resized = image.resize((img_width, img_height))
    
    img_array = np.array(image_resized)
    
    # Jika saat training Anda melakukan normalisasi (/255.0), aktifkan baris di bawah ini:
    img_array = img_array.astype("float32") / 255.0
    
    # Tambahkan dimensi batch (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Prediksi Jenis Berry"):
        with st.spinner("Sedang menganalisis gambar..."):
            predictions = model.predict(img_array)
            
            # Ambil kelas dengan probabilitas tertinggi
            predicted_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_index]
            confidence = float(np.max(predictions[0])) * 100

            # Tampilkan Hasil
            st.success(f"**Hasil Prediksi:** {predicted_class}")
            st.info(f"**Tingkat Kepercayaan (Confidence):** {confidence:.2f}%")
            
            # Menampilkan rincian probabilitas semua kelas (opsional)
            with st.expander("Lihat Detail Probabilitas Semua Kelas"):
                for idx, score in enumerate(predictions[0]):
                    st.write(f"- {class_names[idx]}: `{score*100:.2f}%`")