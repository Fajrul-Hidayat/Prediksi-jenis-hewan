import streamlit as st
import pandas as pd
import Orange
import pickle
import numpy as np

# --- 1. Load model hasil training di Orange ---
# Pastikan file "zoo.pkcls" berada di folder yang sama dengan file ini
with open("zoo.pkcls", "rb") as f:
    model = pickle.load(f)

# --- 2. Judul Aplikasi ---
st.title("🐾 Prediksi Jenis Hewan - Model Random Forest (Orange3)")

st.write("""
Aplikasi ini menggunakan **model Random Forest dari Orange3** 
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
    # Siapkan input sesuai urutan kolom pada zoo.tab
    input_data = pd.DataFrame([[
        hair, feathers, eggs, milk, airborne, aquatic, predator, toothed,
        backbone, breathes, venomous, fins, legs, tail, domestic, catsize
    ]], columns=[
        "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator",
        "toothed", "backbone", "breathes", "venomous", "fins", "legs",
        "tail", "domestic", "catsize"
    ])

    try:
        # Ambil domain model
        model_domain = model.domain

        # --- Buat domain hanya berisi atribut (tanpa class) ---
        # Ini penting supaya Orange tidak menuntut kolom class pada input baru
        feature_domain = Orange.data.Domain(model_domain.attributes)

        # Debug cepat: periksa jumlah atribut cocok
        expected_attr_names = [a.name for a in feature_domain.attributes]
        provided_attr_names = list(input_data.columns)
        if expected_attr_names != provided_attr_names:
            st.warning("Urutan/daftar atribut input tidak sama persis seperti yang diharapkan model.")
            st.write("Atribut model yang diharapkan:", expected_attr_names)
            st.write("Atribut yang diberikan:", provided_attr_names)
            # Lanjutkan tetap mencoba jika user setuju (di sini kita lanjut otomatis)

        # Konversi ke numpy float
        X = input_data.values.astype(float)

        # Buat Orange Table dari feature_domain (tanpa class)
        orange_data = Orange.data.Table.from_numpy(feature_domain, X)

        # Prediksi menggunakan model Orange
        preds = model(orange_data)        # ini mengembalikan prediksi untuk tiap baris
        prediction = preds[0]            # ambil prediksi baris pertama

        # Dapatkan nama label dari model.domain.class_var
        # Jika model mengembalikan Distribution/nomor, ambil index -> nama
        try:
            # prediction mungkin berupa angka indeks atau array; usaha cast ke int
            pred_index = int(prediction)
            label_name = model_domain.class_var.values[pred_index]
        except Exception:
            # kalau prediction adalah object lain (mis. Distribution), coba method paling umum
            try:
                label_name = str(prediction)
            except Exception:
                label_name = "Unknown"

        st.subheader("🎯 Hasil Prediksi:")
        st.success(f"Hewan ini kemungkinan besar adalah **{label_name}**.")

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
        st.write("**Debug domain model (untuk developer):**")
        st.write(model.domain)
        st.write("**Atribut domain model:**")
        st.write([a.name for a in model.domain.attributes])
        st.write("**class_var (target) model:**")
        st.write(model.domain.class_var)
        st.write("**Input (DataFrame)**:")
        st.write(input_data)
