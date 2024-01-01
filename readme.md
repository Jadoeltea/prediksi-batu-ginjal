# Laporan Proyek Machine Learning
### Nama  : **Irfan Zulkarnaen**
### Nim   : 231352002
### Kelas : Malam B

# Aplikasi Prediksi Batu Ginjal
### **Summary**

Aplikasi prediksi batu ginjal adalah sebuah program komputer yang menggunakan informasi tentang kesehatan seseorang, seperti riwayat medis dan hasil tes, untuk memperkirakan risiko pembentukan batu ginjal. Dengan menganalisis data ini, aplikasi memberikan perkiraan tentang seberapa besar kemungkinan seseorang mengalami masalah dengan batu ginjal di masa depan. Hal ini membantu dokter untuk memberikan saran pencegahan atau perawatan lebih awal kepada individu yang berisiko tinggi.    


### **Business Understanding**
Pengembangan aplikasi ini mungkin merupakan bagian dari inovasi lebih luas dalam teknologi kesehatan. Hal ini dapat mencakup penggunaan machine learning atau kecerdasan buatan untuk menghasilkan prediksi yang lebih akurat dan memperluas jangkauan penggunaan aplikasi untuk prediksi penyakit lainnya.


### **Goals**
Dengan fokus pada deteksi dini, pencegahan, perawatan yang lebih tepat, serta pengurangan beban pada sistem kesehatan, aplikasi prediksi batu ginjal bertujuan untuk membantu individu dan menyumbangkan kemajuan dalam sektor kesehatan melalui teknologi yang inovatif.


### **Solution statements**
Saya akan mencoba mengembangkan sebuah aplikasi menggunakan bahasa pemrograman Python  untuk memprediksi penyakit batu Ginjal.

### **Data Understanding**
Dalam proyek ini, saya menggunakan dataset dari [Kaggle](https://www.kaggle.com). Dataset ini, yang disebut [Chronic KIdney Disease dataset](https://www.kaggle.com/datasets/mansoordaku/ckdisease?resource=download)
, berisi data kesehatan yang kita gunakan untuk menganalisis dan memprediksi penyakit batu ginjal. dataset [Chronic KIdney Disease dataset](https://www.kaggle.com/datasets/mansoordaku/ckdisease?resource=download) ini dipilih karena relevansi atribut-atributnya dalam proyek ini."

###  **Variabel-variabel pada Aplikasi Data Mining dalam memprediksi Penyakit Jantung menggunakan Python** 
Selanjutnya variabel atau fitur pada data ini adalah sebagai berikut :  

    -) age/umur (umur dalam tahun)
    -) Blood Pressure / bp (in mm/Hg)
    -) Specific Gravity / sg - (1.005,1.010,1.015,1.020,1.025)
    -) Albumin / al - (0,1,2,3,4,5)
    -) Sugar / su - (0,1,2,3,4,5)
    -) Red Blood Cells / rbc - (normal,abnormal)
    -) Pus Cell / pc - (normal,abnormal)
    -) Pus Cell clumps / pcc - (present,notpresent)
    -) Bacteria / ba  - (present,notpresent)
    -) Blood Glucose Random / bgr in mgs/dl
    -) Blood Urea  / bu (in mgs/dl)
    -) Serum Creatinine / sc (in mgs/dl)
    -) Sodium / sod (in mEq/L)
    -) Potassium / pot (in mEq/L)
    -) Hemoglobin / hemo (in gms)
    -) Packed  Cell Volume /pcv
    -) White Blood Cell Count / wc (in cells/cumm)
    -) Red Blood Cell Count / rc (in millions/cmm)
    -) Hypertension / htn (yes,no)
    -) Diabetes Mellitus / dm (yes,no)
    -) Coronary Artery Disease / cad (yes,no)
    -) Appetite / appet (good,poor)
    -) Pedal Edema / pe (yes,no)	
    -) Anemia / ane (yes,no)
    -) Classification / class (ckd,notckd)

Adapun tipe data dalam dataset [Chronic KIdney Disease dataset](https://www.kaggle.com/datasets/mansoordaku/ckdisease?resource=download) yaitu:

 No|  Column  |  Non-Null Count | Dtype  |
---| ---------| ----------------|--------| 
 0  | age            | 391 non-null  |  float64
 1  | bp             | 388 non-null  |  float64
 2  | sg             | 353 non-null  |  float64
 3  | al             | 354 non-null  |  float64
 4  | su             | 351 non-null  |  float64
 5  | rbc            | 248 non-null  |  object 
 6  | pc             | 335 non-null  |  object 
 7  | pcc            | 396 non-null  |  object 
 8  | ba             | 396 non-null  |  object 
 9  | bgr            | 356 non-null  |  float64
 10 | bu             | 381 non-null  |  float64
 11 | sc             | 383 non-null  |  float64
 12 | sod            | 313 non-null  |  float64
 13 | pot            | 312 non-null  |  float64
 14 | hemo           | 348 non-null  |  float64
 15 | pcv            | 330 non-null  |  object 
 16 | wc             | 295 non-null  |  object 
 17 | rc             | 270 non-null  |  object 
 18 | htn            | 398 non-null  |  object 
 19 | dm             | 398 non-null  |  object 
 20 | cad            | 398 non-null  |  object 
 21 | appet          | 399 non-null  |  object 
 22 | pe             | 399 non-null  |  object 
 23 | ane            | 399 non-null  |  object 
 24 | classification |400 non-null   |  object 
dtypes: float64(11), object(14)

**Visualisasi Data**:
- Dalam Melakukan pemahaman terhadap dataset [Chronic KIdney Disease dataset](https://www.kaggle.com/datasets/mansoordaku/ckdisease?resource=download), saya membuat 6 grafik  yang berbeda dimana, pada grafik yang pertama menunjukkan jumlah penderita penyakit Pola Sebaran Penderita Penyakit Ginjal Kronis (CKD) Berdasarkan Usia Dengan Diabetes Melitus (DM) , yang kedua jumlah penderita berdasarkan Pola Sebaran Penderita Penyakit Ginjal Kronis (CKD) Berdasarkan Usia Dengan Hypertensiom (htn), Pola Sebaran Penderita Penyakit Ginjal Kronis (CKD) Berdasarkan Usia Dengan Anemia (ane),Distribusi Frekuensi Data dari Kolom Numerik, Heatmap Korelasi antar Kolom Numerik, Proporsi Nilai dalam Kolom Kategorikal.
- Dan terakhir dalam memvisualisasikan modelling menggunakan algoritma ID3 atau pohon keputusan  


## Data Preparation
Teknik data preparation yang dilakukan adalah :
- Import Dataset di Kaggle :
```
import os
import json

kaggle_json = {
    "username": "jadoeltea",
    "key": "e4829efe858500918d02c29ecf585080"
}

kaggle_dir = '/root/.kaggle'
kaggle_file = 'kaggle.json'
kaggle_path = os.path.join(kaggle_dir, kaggle_file)

if not os.path.exists(kaggle_dir):
    os.makedirs(kaggle_dir)


with open(kaggle_path, 'w') as file:
    json.dump(kaggle_json, file)

os.chmod(kaggle_path, 0o600)
```
```
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
!kaggle datasets download -d mansoordaku/ckdisease
!unzip ckdisease.zip
```
- Import Library yang akan digunakan :
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pickle
```
- Data Discovery :
```
data = pd.read_csv('kidney_disease.csv')
data.sample()
```
|index|id|age|bp|sg|al|su|rbc|pc|pcc|ba|bgr|bu|sc|sod|pot|hemo|pcv|wc|rc|htn|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|206|206|60\.0|70\.0|1\.01|1\.0|0\.0|NaN|normal|notpresent|notpresent|109\.0|96\.0|3\.9|135\.0|4\.0|13\.8|41|NaN|NaN|yes|
```
data = data.drop(['id'], axis=1)
data.info()
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 400 entries, 0 to 399
 No |  Column  |  Non-Null Count | Dtype       |
--- | ---------| ----------------|--------     | 
 0  | age      | 391 non-null  |  float64
 1  | bp       | 388 non-null  |  float64
 2  | sg       | 353 non-null  |  float64
 3  | al       | 354 non-null  |  float64
 4  | su       | 351 non-null  |  float64
 5  | rbc      | 248 non-null  |  object 
 6  | pc       | 335 non-null  |  object 
 7  | pcc      | 396 non-null  |  object 
 8  | ba       | 396 non-null  |  object 
 9  | bgr      | 356 non-null  |  float64
 10 | bu       | 381 non-null  |  float64
 11 | sc       | 383 non-null  |  float64
 12 | sod      | 313 non-null  |  float64
 13 | pot      | 312 non-null  |  float64
 14 | hemo     | 348 non-null  |  float64
 15 | pcv      | 330 non-null  |  object 
 16 | wc       | 295 non-null  |  object 
 17 | rc       | 270 non-null  |  object 
 18 | htn      | 398 non-null  |  object 
 19 | dm       | 398 non-null  |  object 
 20 | cad      | 398 non-null  |  object 
 21 | appet    | 399 non-null  |  object 
 22 | pe       | 399 non-null  |  object 
 23 | ane      | 399 non-null  |  object 
 24 | classification |400 non-null   |  object 
dtypes: float64(11), object(14)
memory usage: 81.4+ KB
```
numerical = []
catgcols = []

for col in data.columns:
  if data[col].dtype=="float64":
    numerical.append(col)
  else:
      catgcols.append(col)

for col in data.columns:
     if col in numerical:
        data[col].fillna(data[col].median(), inplace=True)
     else:
        data[col].fillna(data[col].mode()[0], inplace=True)
  ```
```
numerical
```
['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo']
```
catgcols
```
['rbc',
 'pc',
 'pcc',
 'ba',
 'pcv',
 'wc',
 'rc',
 'htn',
 'dm',
 'cad',
 'appet',
 'pe',
 'ane',
 'classification']

- Visualisasi Data (EDA)
```
  fig = px.scatter(data,
    x = data['age'], y = data['dm'],
    color="classification")
fig.update_layout(title='Pola Sebaran Penderita Penyakit Ginjal Kronis (CKD) Berdasarkan Usia Dengan Diabetes Melitus (DM)')
fig.show()
```
![alt text](https://github.com/Jadoeltea/prediksi-batu-ginjal/blob/main/public/1.png?raw=true)

```
fig = px.scatter(data,
    x = data['age'], y = data['htn'],
    color="classification")
fig.update_layout(title='Pola Sebaran Penderita Penyakit Ginjal Kronis (CKD) Berdasarkan Usia Dengan Hypertensiom (htn)')
fig.show()
```
![alt text](https://github.com/Jadoeltea/prediksi-batu-ginjal/blob/main/public/2.png?raw=true)
```
fig = px.scatter(data,
    x = data['age'], y = data['ane'],
    color="classification")
fig.update_layout(title='Pola Sebaran Penderita Penyakit Ginjal Kronis (CKD) Berdasarkan Usia Dengan Anemia (ane)')
fig.show()
```
![alt text](https://github.com/Jadoeltea/prediksi-batu-ginjal/blob/main/public/3.png?raw=true)
```
#distribusi frekuensi dari kolom numerik
plt.figure(figsize=(20, 15))
plotnumber = 1

for column in numerical:
    if plotnumber <= 14:
        ax = plt.subplot(3, 5, plotnumber)
        sns.histplot(data[column], kde=True)
        plt.xlabel(column)

    plotnumber += 1
plt.suptitle('Distribusi Frekuensi Data dari Kolom Numerik', fontsize=16)
plt.tight_layout()
plt.show()
```
![alt text](https://github.com/Jadoeltea/prediksi-batu-ginjal/blob/main/public/4.png?raw=true)
```
numeric_columns = data.select_dtypes(include=['float64']).columns

plt.figure(figsize=(10, 8))
sns.heatmap(data[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap Korelasi antar Kolom Numerik')
plt.show()
```
![alt text](https://github.com/Jadoeltea/prediksi-batu-ginjal/blob/main/public/5.png?raw=true)
```
plt.figure(figsize=(12, 8))

for i, column in enumerate(catgcols, 1):
    plt.subplot(4, 4, i)
    value_counts = data[column].value_counts()
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
    plt.title(column)
plt.suptitle('Proporsi Nilai dalam Kolom Kategorikal', fontsize=16)
plt.tight_layout()
plt.show()
```
![alt text](https://github.com/Jadoeltea/prediksi-batu-ginjal/blob/main/public/6.png?raw=true)

- Data Preprocesing
```
data['classification'].value_counts()
```
ckd       248
notckd    150
ckd\t       2
Name: classification, dtype: int64

```
data['classification'] =data ['classification'].replace(['ckd\t'],'ckd')
```

```
data['classification'].value_counts()
```
ckd       250
notckd    150
Name: classification, dtype: int64

```
#memisahkan features dan label
ind_col = [col for col in data.columns if col != 'classification']
dep_col = 'classification'
```

```
data[dep_col].value_counts()
```
ckd       250
notckd    150
Name: classification, dtype: int64
```
#merubah dataset semua menjadi numerikal
le = LabelEncoder()
for col in catgcols:
  data[col] = le.fit_transform(data[col])
```

```
data['classification'] = le.fit_transform(data['classification'])
```

```
#identifikasi data label dan features
x = data[ind_col]
y = data[dep_col]
```

```
data.head()
```
|index|age|bp|sg|al|su|rbc|pc|pcc|ba|bgr|bu|sc|sod|pot|hemo|pcv|wc|rc|htn|dm|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|48\.0|80\.0|1\.02|1\.0|0\.0|1|1|0|0|121\.0|36\.0|1\.2|138\.0|4\.4|15\.4|32|72|34|1|4|
|1|7\.0|50\.0|1\.02|4\.0|0\.0|1|1|0|0|121\.0|18\.0|0\.8|138\.0|4\.4|11\.3|26|56|34|0|3|
|2|62\.0|80\.0|1\.01|2\.0|3\.0|1|1|0|0|423\.0|53\.0|1\.8|138\.0|4\.4|9\.6|19|70|34|0|4|
|3|48\.0|70\.0|1\.005|4\.0|0\.0|1|0|1|0|117\.0|56\.0|3\.8|111\.0|2\.5|11\.2|20|62|19|1|3|
|4|51\.0|80\.0|1\.01|2\.0|0\.0|1|1|0|0|106\.0|26\.0|1\.4|138\.0|4\.4|11\.6|23|68|27|0|3|
```
data.to_csv('kidney-disease.csv')
```
- Modelling
```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
```

```
dtc = DecisionTreeClassifier (
    ccp_alpha=0.0, class_weight=None, criterion='entropy',
    max_depth=4, max_features=None, max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_samples_leaf=1,
    min_samples_split=2, min_weight_fraction_leaf=0.0,
    random_state=42, splitter='best'
    )

model = dtc.fit(x_train, y_train)
dtc_acc = accuracy_score(y_test, dtc.predict(x_test))

print(f"akurasi data training = {accuracy_score(y_train, dtc.predict(x_train))}")
print(f"akurasi data testing = {dtc_acc} \n")

print(f"confusion matrix : \n{confusion_matrix(y_test, dtc.predict(x_test))}")
confusion = confusion_matrix (y_test, dtc.predict(x_test))

tn, fp, fn, tp = confusion.ravel()
print (f"classification report : \n {classification_report(y_test, dtc.predict(x_test))}")
```
akurasi data training = 0.99375
akurasi data testing = 1.0 

confusion matrix : 
[[52  0]
 [ 0 28]]
classification report : 
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        52
           1       1.00      1.00      1.00        28

    accuracy                           1.00        80
   macro avg       1.00      1.00      1.00        80
weighted avg       1.00      1.00      1.00        80

- Visualisasi hasil Algoritma ID3 / Pohon Keputusan
```
fig = plt.figure(figsize=(30,25))
_= tree.plot_tree(model,
                 feature_names=ind_col,
                 class_names=['notckd','ckd'],
                 filled=True)
```
![alt text](https://github.com/Jadoeltea/prediksi-batu-ginjal/blob/main/public/7.png?raw=true)

```
input_data = (68.0,70.0,1.005,1.0,0.0,0,0,1,0,121.0,28.0,1.4,138.0,4.4,12.9,26,90,34,0,3, 2, 0, 0,0)
input_data_as_numpy_array = np.array(input_data)
input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

# Melakukan prediksi dengan model yang sudah diinisiasi sebelumnya
prediction = model.predict(input_data_reshape)
print(prediction)

if prediction[0] == 0:
    print('Pasien Tidak Terkena Batu Ginjal')
else:
    print('Pasien Terkena Batu Ginjal')
```
[0]
Pasien Tidak Terkena Batu Ginjal

/usr/local/lib/python3.10/dist-packages/sklearn/base.py:439: UserWarning:

X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names


- Simpan model untuk proses deploy ke streamlit
```
filename = 'kidney-disease.sav'
pickle.dump(dtc, open(filename,'wb'))
```

## Modeling
Untuk mengimplementasikan algoritma ID3 atau pohon keputusan secara matematis untuk dataset Chronic Kidney Disease (CKD), rumus yang digunakan adalah untuk menghitung entropy dan information gain.

1. Entropy:

Entropy dari sebuah set data dapat dihitung menggunakan rumus:

![alt text](https://github.com/Jadoeltea/prediksi-batu-ginjal/blob/main/public/9.PNG?raw=true)

2. Information Gain:

Information gain untuk sebuah atribut dalam dataset dapat dihitung sebagai perbedaan antara entropy dataset keseluruhan sebelum dan setelah pemisahan berdasarkan atribut tersebut. Rumusnya adalah:

![alt text](https://github.com/Jadoeltea/prediksi-batu-ginjal/blob/main/public/10.PNG?raw=true)


Dengan menggunakan rumus-rumus tersebut, langkah-langkah ID3 untuk membangun pohon keputusan melibatkan perhitungan entropy dan information gain untuk setiap atribut dalam dataset CKD. Atribut dengan information gain tertinggi dipilih sebagai atribut pembagi pada setiap level pohon keputusan, dan proses ini diulang hingga terbentuk pohon keputusan yang lengkap.

## Evaluation
Hasil dari evaluasi model prediksi penyakit ginjal berdasarkan data training dan data testing menunjukkan kinerja yang sangat baik. Model yang dikembangkan memiliki tingkat akurasi yang tinggi pada kedua dataset, dengan akurasi data training sebesar 99.375% dan akurasi data testing mencapai 100%.

Dilihat dari confusion matrix, model berhasil memprediksi semua sampel pada data testing dengan benar. Tidak ada sampel yang salah diklasifikasikan, dengan 52 sampel dari kelas 0 dan 28 sampel dari kelas 1 semuanya terprediksi dengan tepat.

Dalam classification report, hasil evaluasi model menunjukkan bahwa model memiliki precision, recall, dan f1-score sebesar 1.00 untuk kedua kelas. Hal ini menandakan bahwa model dapat memprediksi kedua kelas (CKD dan non-CKD) dengan sempurna.

Secara keseluruhan, hasil evaluasi model menunjukkan performa yang sangat baik dengan nilai akurasi yang tinggi dan hasil evaluasi lainnya yang mencapai nilai maksimal. Ini menunjukkan bahwa model memiliki kemampuan yang sangat baik dalam memprediksi penyakit ginjal berdasarkan atribut-atribut yang ada dalam dataset.


## Deployment
Berikut hasil dari akhir proyek [APLIKASI PREDIKSI PENYAKIT BATU GINJAL](https://s.id/AppPredictBatuGinjal)
![alt text](https://github.com/Jadoeltea/prediksi-batu-ginjal/blob/main/halaman%20prediksi.PNG?raw=true)





**---Ini adalah bagian akhir laporan---**



