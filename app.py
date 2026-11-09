import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
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

# Deteksi ukuran input dari model secara otomatis
input_shape = model.input_shape
if input_shape and len(input_shape) >= 3:
    img_height = input_shape[1] if input_shape[1] is not None else 150
    img_width = input_shape[2] if input_shape[2] is not None else 150
else:
    img_height, img_width = 150, 150

# Antarmuka Pengguna (UI)
st.title("🍓 Aplikasi Klasifikasi Citra Berry")
st.markdown("Aplikasi web berbasis **Convolutional Neural Network (CNN)** untuk mendeteksi jenis buah berry secara otomatis.")
st.markdown("---")

# Sidebar untuk informasi tambahan
with st.sidebar:
    st.markdown("---")
    st.subheader("📋 Kelas Model:")
    for name in class_names:
        st.code(name)

# Unggah Gambar
uploaded_file = st.file_uploader("Pilih atau seret gambar buah berry di sini...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    
    # Tampilkan gambar dengan tata letak yang rapi
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
    
    with col2:
        st.info("💡 **Tips Pengujian:**\nPastikan gambar buah berry terlihat jelas agar model dapat memberikan tingkat akurasi (confidence) yang tinggi.")
    
    # Pra-pemrosesan gambar
    image_resized = image.resize((img_width, img_height))
    img_array = np.array(image_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Jalankan Prediksi", type="primary", use_container_width=True):
        with st.spinner("Model sedang menganalisis pola gambar..."):
            try:
                predictions = model.predict(img_array)
                
                predicted_index = np.argmax(predictions[0])
                predicted_class = class_names[predicted_index]
                confidence = float(np.max(predictions[0])) * 100

                st.markdown("---")
                st.subheader("🎯 Hasil Analisis Model")
                
                # Kotak hasil menonjol
                st.success(f"**Jenis Berry Terdeteksi:** `{predicted_class}`")
                st.metric(label="Tingkat Kepercayaan (Confidence)", value=f"{confidence:.2f}%")
                
                # Visualisasi Grafik Batang (Bar Chart) untuk semua kelas
                st.markdown("### 📊 Grafik Probabilitas Semua Kelas")
                chart_data = pd.DataFrame({
                    'Kelas Berry': class_names,
                    'Probabilitas': predictions[0] * 100
                }).set_index('Kelas Berry')
                
                st.bar_chart(chart_data)
                        
            except Exception as e:
                st.error(f"Terjadi Kesalahan saat prediksi: `{str(e)}`")
