import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. Load model hasil training (tanpa Orange) ---
# Pastikan file zoo.pkcls adalah model scikit-learn
with open("zoo.pkcls", "rb") as f:
    model = pickle.load(f)

# --- 2. Judul Aplikasi ---
st.title("🐾 Prediksi Jenis Hewan")

st.write("""
Aplikasi ini menggunakan **model Random Forest** 
untuk memprediksi **jenis hewan (type)** berdasarkan karakteristik fisik.
Silakan isi form di bawah ini:
""")

# --- 3. Fungsi Input (Yes/No → 1/0) ---
def yes_no(label):
    return 1 if st.selectbox(label, ["No", "Yes"]) == "Yes" else 0

# --- 4. Input dari pengguna ---
hair = yes_no("Apakah memiliki rambut (hair)?")
feathers = yes_no("Apakah memiliki bulu (feathers)?")
eggs = yes_no("Apakah bertelur (eggs)?")
milk = yes_no("Apakah menghasilkan susu (milk)?")
airborne = yes_no("Apakah bisa terbang (airborne)?")
aquatic = yes_no("Apakah hidup di air (aquatic)?")
predator = yes_no("Apakah predator?")
toothed = yes_no("Apakah memiliki gigi (toothed)?")
backbone = yes_no("Apakah memiliki tulang belakang (backbone)?")
breathes = yes_no("Apakah bernapas dengan paru-paru (breathes)?")
venomous = yes_no("Apakah beracun (venomous)?")
fins = yes_no("Apakah memiliki sirip (fins)?")
legs = st.selectbox("Jumlah kaki (legs):", [0, 2, 4, 6, 8])
tail = yes_no("Apakah memiliki ekor (tail)?")
domestic = yes_no("Apakah hewan peliharaan (domestic)?")
catsize = yes_no("Apakah berukuran seperti kucing (catsize)?")

# --- 5. Tombol Prediksi ---
if st.button("🔍 Prediksi Jenis Hewan"):
    # Siapkan input
    input_data = np.array([[hair, feathers, eggs, milk, airborne, aquatic, predator,
                            toothed, backbone, breathes, venomous, fins, legs,
                            tail, domestic, catsize]])

    # Prediksi
    try:
        pred = model.predict(input_data)[0]

        st.subheader("🎯 Hasil Prediksi:")
        st.success(f"Hewan ini kemungkinan besar adalah **{pred}**.")

        st.write("### 📘 Keterangan Kelas:")
        st.write({
            "mammal": "Mamalia",
            "bird": "Burung",
            "reptile": "Reptil",
            "fish": "Ikan",
            "amphibian": "Amfibi",
            "insect": "Serangga",
            "invertebrate": "Tanpa tulang belakang"
        })

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
