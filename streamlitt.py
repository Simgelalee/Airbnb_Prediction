import streamlit as st
import pandas as pd
import joblib

# Streamlit Sayfa Yapılandırması
st.set_page_config(layout="wide", page_title="NYC Emlak Fiyat Tahmini", page_icon="🏠")

# Veri ve Model Yükleme
@st.cache_data
def veri_yukle():
    return pd.read_csv(r"C:\Users\ASUS\PycharmProjects\pythonProject\Airbnb_prediction_Son\yeni_dosya1.csv")

@st.cache_resource
def model_yukle():
    return joblib.load(r"C:\Users\ASUS\PycharmProjects\pythonProject\Airbnb_prediction_Son\yeni_cikti_catboost.pkl")

df = veri_yukle()
model = model_yukle()

# Başlık ve Açıklama
st.title("🏠 NYC Emlak Fiyat Tahmini")
st.subheader("Bölge grubu, oda tipi ve minimum gece bilgisiyle tahmini fiyatı öğrenin!")

# Kullanıcıdan Bilgi Alma
st.write("### Mülk Detaylarını Girin")

# Bölge Grubu Seçimi
bolge_grubu = st.selectbox(
    "Bölge grubunuzu seçin:",
    options=["Brooklyn", "Manhattan", "Queens", "Staten Island"],
    help="Mülkün bulunduğu genel bölgeyi seçin (örn: Manhattan, Brooklyn)."
)

# Oda Tipi Seçimi
oda_tipi = st.selectbox(
    "Oda tipini seçin:",
    options=["Entire home/apt", "Private room", "Shared room"],
    help="Mülkün oda tipini seçin."
)

# Minimum Gece Sayısı
min_gece = st.number_input(
    "Minimum gece sayısını girin:",
    min_value=1,
    max_value=365,
    value=1,
    step=1,
    help="Misafirlerin en az kaç gece kalması gerektiğini belirtin."
)

# Tahmin Butonu
if st.button("Tahmini Fiyatı Hesapla"):
    # Kullanıcı girdilerini işleme
    input_data = {
        "minimum_nights": min_gece,
        "room_type_Private room": 1 if oda_tipi == "Private room" else 0,
        "room_type_Shared room": 1 if oda_tipi == "Shared room" else 0,
    }

    # Bölge Grubu Dummy Encoding (tek bir 1, diğerleri 0 olacak şekilde)
    for group in ["Brooklyn", "Manhattan", "Queens", "Staten Island"]:
        input_data[f"neighbourhood_group_{group}"] = 1 if bolge_grubu == group else 0

    # Kullanıcı girdisini DataFrame'e dönüştür
    input_df = pd.DataFrame([input_data])

    # Modelin eğitimde kullandığı sütunları al
    feature_names = model.feature_names_

    # Eksik sütunları tamamlayarak, yalnızca modelin beklediği sütunlarla eşleşen bir DataFrame oluştur
    for col in feature_names:
        if col not in input_df:
            input_df[col] = 0  # Eksik sütunlara varsayılan 0 değeri ekleniyor

    # Fazla sütunları kaldır
    input_df = input_df[feature_names]

    # Dummy Encoding Kontrolü
    st.write("Dummy Encoding Sonuçları:")
    st.write(input_df)

    # Tahmin işlemi
    tahmini_fiyat = model.predict(input_df)[0]

    # Sonuç Gösterimi
    st.success(f"Tahmini Fiyat: *{tahmini_fiyat:.2f} USD*")