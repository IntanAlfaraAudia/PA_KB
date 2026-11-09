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

# Deteksi ukuran input dari model
input_shape = model.input_shape
if input_shape and len(input_shape) >= 3:
    img_height = input_shape[1] if input_shape[1] is not None else 150
    img_width = input_shape[2] if input_shape[2] is not None else 150
else:
    img_height, img_width = 150, 150

# Antarmuka Pengguna (UI)
st.title("🍓 Aplikasi Klasifikasi Citra Berry")
st.write("Unggah gambar buah berry untuk mendeteksi jenisnya secara otomatis menggunakan model Convolutional Neural Network (CNN).")

uploaded_file = st.file_uploader("Pilih gambar buah berry (format JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
    
    image_resized = image.resize((img_width, img_height))
    
    # Ubah ke numpy array tanpa dibagi 255 (karena model sudah menangani rescaling sendiri jika ada layer Rescaling)
    img_array = np.array(image_resized, dtype=np.float32)
    
    # Tambahkan dimensi batch (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Prediksi Jenis Berry"):
        with st.spinner("Sedang menganalisis gambar..."):
            try:
                # Tampilkan info debug di web untuk memastikan bentuknya
                st.info(f"Debug -> Model Input Shape: {model.input_shape} | Image Shape: {img_array.shape}")
                
                predictions = model.predict(img_array)
                
                predicted_index = np.argmax(predictions[0])
                predicted_class = class_names[predicted_index]
                confidence = float(np.max(predictions[0])) * 100

                st.success(f"**Hasil Prediksi:** {predicted_class}")
                st.info(f"**Tingkat Kepercayaan (Confidence):** {confidence:.2f}%")
                
                with st.expander("Lihat Detail Probabilitas Semua Kelas"):
                    for idx, score in enumerate(predictions[0]):
                        st.write(f"- {class_names[idx]}: `{score*100:.2f}%`")
                        
            except Exception as e:
                st.error(f"Terjadi Error saat prediksi: `{str(e)}`")
