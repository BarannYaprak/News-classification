

import numpy as np
from sentence_transformers import SentenceTransformer

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download("punkt")

df_veri = pd.read_json("/content/drive/MyDrive/dogal_dil/proje/news.json")
veri = df_veri["sentence"]

df["temiz_veri_2"] = veri.apply(lambda x: re.sub(r'#\w+', '',x))
df["temiz_veri_2"] = veri.apply(lambda x: re.sub(r"http\S+","",x))
df["temiz_veri_2"] = veri.apply(lambda x: re.sub(r"\d+","",x))

vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(df["temiz_veri_2"])



from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import json

X = vectors
y = df["label"]

def sonuclar(model,X_test,y_test,model_name):
    # Modelin tahminlerinin yapılma aşaması
    y_pred = model.predict(X_test)

    # Doğruluk (accuracy) hesaplanma aşaması
    accuracy = accuracy_score(y_test, y_pred)

    # Hassasiyet (precision) hesaplanma aşaması
    precision = precision_score(y_test, y_pred, average= "macro")

    # Geri çağırma (recall) hesaplanma aşaması
    recall = recall_score(y_test, y_pred, average= "macro")

    # F1 skoru hesaplanma aşaması
    f1 = f1_score(y_test, y_pred, average= "macro")

    # Karar matrisini hesaplanma aşaması
    cm = confusion_matrix(y_test, y_pred)

    # TP, FP, TN ve FN sayılarınının hesaplanma aşaması
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]

    # TP, FP, TN ve FN oranlarınıın hesaplanma aşaması
    TP_rate = TP / (TP + FN)
    FP_rate = FP / (FP + TN)
    TN_rate = TN / (TN + FP)
    FN_rate = FN / (FN + TP)


    # Metrikleri sözlükte toplanması
    metrics = {'accuracy': [accuracy],
            'precision': [precision],
            'recall': [recall],
            'f1': [f1],
            'TP_rate': [TP_rate],
            'FP_rate': [FP_rate]}


    print(metrics)
    print(cm)

def egitim(metot,model_name):
    # Veri setini train ve test olarak bölünmesi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle= True)
    #modelin eğitimi
    model = metot
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    #Test sonuçlarının yazdırılması
    print(f"Test score: {score}")
    sonuclar(model,X_test,y_test,model_name)
    print("\n")

model = DecisionTreeClassifier()
egitim(model,"Karar Ağacı")

model_nnm = MLPClassifier(batch_size = 1000)
egitim(model_nnm,"Yapay sinir ağı")

model_log = LogisticRegression(random_state=0)
egitim(model_log,"lojistik regresyon")



model = SentenceTransformer('ytu-ce-cosmos/turkish-base-bert-uncased')

import pandas as pd
import csv
dosya_df = pd.read_json('/content/drive/MyDrive/dogal_dil/proje/news.json')
dosya_df

temsiller= []

for i in range(len(dosya_df)):
  text = dosya_df.iloc[i]["sentence"]
  vector = model.encode(text, show_progress_bar=False)
  temsiller.append(vector)
  print(i)

df = pd.DataFrame(temsiller)

# Excel dosyası olarak kaydet
df.to_excel('/content/drive/MyDrive/dogal_dil/proje/temsiller.xlsx', index=True)

X = temsiller
y = dosya_df["label"]

print(len(X))
print(len(y))

from sklearn.naive_bayes import GaussianNB
model_gau = GaussianNB()
egitim(model_gau,"Naive Bayes")



model_log = LogisticRegression(random_state=0, max_iter= 20000)
egitim(model_log,"lojistik regresyon")



model_nnm = MLPClassifier(batch_size = 1000, max_iter =1000)
egitim(model_nnm,"Yapay sinir ağı")

from sklearn.neighbors import KNeighborsClassifier

# KNN modelini oluştur
knn_model = KNeighborsClassifier(n_neighbors=3)
egitim(knn_model,"K- en yakın komşu")


def model_kaydet(model,model_name):
    import pickle
    with open(f'/content/drive/MyDrive/dogal_dil/proje/{model_name}.pickle', 'wb') as f:
      pickle.dump(model, f)

model_kaydet(model_log, "lojistik regresyon 20000 iter")

def model_oku(model_name):
    import pickle


    with open(f'/content/drive/MyDrive/dogal_dil/proje/{model_name}.pickle', 'rb') as f:
      predict_model = pickle.load(f)
    return predict_model

def vektor_olustur(text):
  #Tahmin yapılacak metinleri anlam vektörlerinin oluşturulması
  model = SentenceTransformer('ytu-ce-cosmos/turkish-base-bert-uncased')
  vector = model.encode(text, show_progress_bar=False)
  return vector

def tahmin():
  sınıflar = ['dunya','ekonomi','genel','guncel','kultur-sanat','magazin','planet','saglik','siyaset','spor','teknoloji','turkiye','yasam']
  model = model_oku("lojistik regresyon 20000 iter")
  text = input("Haber Metini Giriniz")
  text = vektor_olustur(text)
  text = text.reshape(1, -1)
  predict = model.predict(text)
  i = predict[0] - 1
  print(sınıflar[i])

"""TCMB'nin 24 Mayıs vadeli TL depo alım ihalesinde hem teklif hem de gerçekleşme tutarı 29 milyar 892 milyon lira oldu.

İhalede, minimum faiz oranı yüzde 49,97, maksimum faiz oranı yüzde 50, ortalama faiz oranı ise yüzde 49,99 olarak ilan edildi.

Bankanın 27 Mayıs vadeli TL depo alım ihalesinde de teklif ve gerçekleşme tutarları 5 milyar 200 milyon lira oldu.

İhalede, minimum faiz oranı yüzde 49,99, maksimum faiz oranı yüzde 50, ortalama faiz oranı ise yüzde 49,99 olarak gerçekleşti."""


tahmin()

"""Kurtuluş İttifakı adına yapılan açıklamaya göre, Namık Kemal Zeybek'in genel başkanlığını yürüttüğü
ATA Parti, Rifat Serdaroğlu'nun genel başkanı olduğu Doğru Parti, Sadettin Tantan'ın
genel başkanlığını yaptığı Yurt Partisi, Vecdet Öz'ün genel başkanı olduğu Adalet Partisi ile Ahmet Yılmaz'ın
genel başkanlığını yürüttüğü Milliyetçi Türkiye Partisi "Kurtuluş İttifakı"nı kurma kararı aldı.
İttifakın sözcülüğüne Milliyetçi Türkiye Partisi Genel Başkanı Ahmet Yılmaz getirildi. """
tahmin()

"""2018 yılında evlenen Justin Bieber ve Hailey Baldwin anne baba oluyor.
Çif ilişkileriyle gündemden düşmüyor. Geçtiğimiz haftalarda ağladığı fotoğrafı paylaşan
ünlü şarkıcı hayranlarının kafasını karıştırmıştı. Paylaştığı fotoğrafla ayrılık iddialarını da beraberinde getiren
şarkıcı,çok geçmeden gerçeği açıkladı. Ünlü şarkıcı baba oluyor, çift hayranlarına müjdeli haberi sosyal medya hesaplarından duyurdu."""
tahmin()