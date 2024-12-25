import pandas as pd
import joblib

# Veri ve Model Yükleme
def veri_yukle():
    return pd.read_csv("yeni_dosya1.csv")

def model_yukle():
    return joblib.load("yeni_cikti_catboost.pkl")

# Kullanıcıdan Bilgi Alma ve Tahmin Yapma
def tahmin_yap(bolge_grubu, oda_tipi, min_gece):
    # Veri ve model yükleme
    df = veri_yukle()
    model = model_yukle()

    # Kullanıcı girdilerini işleme
    input_data = {
        "minimum_nights": min_gece,
        "room_type_Private room": 1 if oda_tipi == "Private room" else 0,
        "room_type_Shared room": 1 if oda_tipi == "Shared room" else 0,
    }

    # Bölge Grubu Dummy Encoding
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

    # Tahmin işlemi
    tahmini_fiyat = model.predict(input_df)[0]

    # Sonuç Gösterimi
    return f"{bolge_grubu} - {oda_tipi}: {tahmini_fiyat:.2f} USD"

# Test için şehirler, oda tipleri ve minimum gece sayısı
bolgeler = ["Brooklyn", "Manhattan", "Queens", "Staten Island"]
oda_tipleri = ["Entire home/apt", "Private room", "Shared room"]
min_gece = 1  # Minimum gece sayısı her zaman 1 olacak

# 10 adet test girişi oluşturma ve tahmin yapma
test_sonuc = []
for bolge in bolgeler:
    for oda_tipi in oda_tipleri:
        sonuc = tahmin_yap(bolge, oda_tipi, min_gece)
        test_sonuc.append(sonuc)

# Sonuçları yazdırma
for sonuc in test_sonuc:
    print(sonuc)