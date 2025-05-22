# Laporan Proyek Machine Learning - Aldi Musneldi

## Domain Proyek

Kanker paru-paru adalah salah satu jenis kanker paling mematikan di dunia. Pada tahun 2020, terjadi 2,2 juta kasus kanker paru-paru di seluruh dunia. Di Indonesia sendiri terdapat 396.914 kasus kanker, di mana 34.783 di antaranya (sekitar 8,8%) merupakan kasus kanker paru-paru [2]. Tingginya angka kejadian tersebut disebabkan oleh kurangnya sistem prediksi atau deteksi dini terhadap risiko kanker paru-paru[[1]](http://ijcs.net/ijcs/index.php/ijcs/article/view/3267).

Untuk dapat memprediksi risiko kanker paru-paru, diperlukan data pasien yang kompleks serta metode yang efisien, akurat, dan dapat digunakan secara luas. Oleh karena itu, dalam proyek ini akan dikembangkan model Machine Learning untuk memprediksi risiko kanker paru-paru berdasarkan data pasien.


## Business Understanding

### Problem Statements
- Bagaimana memprediksi risiko kanker paru-paru berdasarkan data pasien?
- Algoritma Machine Learning apa yang paling efektif untuk prediksi risiko kanker paru-paru?
- Metrik evaluasi apa yang paling tepat untuk menilai performa model prediksi ini?

### Goals
- Membangun model prediksi berbasis Naive Bayes dan K-Nearest Neighbors (KNN) untuk mengklasifikasikan apakah seorang pasien berisiko terkena kanker paru-paru atau tidak.
- Membandingkan performa kedua algoritma Machine Learning untuk menentukan model terbaik.
- Mengevaluasi performa model menggunakan metrik seperti akurasi, precision, recall, dan F1-score, serta menganalisis hasil prediksi untuk mengetahui kelebihan dan kelemahan masing-masing model.

### Solution Statement
Solusi dilakukan dengan pendekatan sebagai berikut:
- Membangun 2 model Machine Learning yaitu Naive Bayes dan K-Nearest Neighbors (KNN).
  * 1. Naive Bayes adalah algoritma klasifikasi yang berdasarkan pada teorema Bayes dengan asumsi bahwa  setiap  atribut  yang  ada  dalam  data  adalahindependen  satu  sama  lain. Algoritma  ini  dapat  digunakan  untuk  mengklasifikasikan  data  ke  dalam  kategori atau kelas yang relevan berdasarkan pada probabilitasnya[[2]](http://ijcs.net/ijcs/index.php/ijcs/article/view/3227/178).
      ![image](https://github.com/user-attachments/assets/ab462c68-6cf1-495f-b525-a3fa204e87ac)

      source:[[3]](https://course-net.com/blog/metode-naive-bayes/8).
  * 2. K-Nearest Neighbors (KNN) memiliki  prinsip  kerja  dengan cara  mencari  jarak  paling  dekat  diantara  data  yang akan dievaluasi dengan tetangga yang ada pada data pelatihan,algoritma  ini  merupakan  salah  satu algoritma    yang    sederhana    untuk    memecahkan klasifikasi  serta  mampu  memberikan  hasil  yang signifikan dan kompetitif[[4]](https://www.journal.sekawan-org.id/index.php/jtim/article/view/178/130).
      ![image](https://github.com/user-attachments/assets/e5dfde92-5be1-448b-a251-8705ffec9c0f)

      source: [[5]](https://www.appliedaicourse.com/blog/knn-algorithm-in-machine-learning/)

## Data Understanding
Sumber Data: Dataset diperoleh dari [[Kaggle]](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link)

Struktur Dataset: 


![image](https://github.com/user-attachments/assets/0969917d-fd80-4e95-908f-e9f50a988787)


Dataset ini memiliki beberapa kolom utama yang merepresentasikan informasi data pasien. 
Berikut penjelasan tiap kolom:
| Kolom                  | Tipe Data | Deskripsi                                                                 |
|------------------------|-----------|---------------------------------------------------------------------------|
| `index`                 | int64    | Nomor urut atau indeks baris data                                                  |
| `Patient Id`                 | object    | ID unik untuk mengidentifikasi setiap pasien                                                 |
| `Age`                 | int64    | Usia pasien                                                   |
| `Gender`             | int64   | Jenis kelamin pasien                                          |
| `Air Pollution`           | int64   | Tingkat paparan polusi udara pasien                                                 |
| `Alcohol use`          | int64   | Tingkat penggunaan alkohol pasien                                         |
| `Dust Allergy`                 | int64   | Tingkat alergi debu pasien                                                 |
| `OccuPational Hazards`                  | int64   | Tingkat bahaya pekerjaan pasien                                                  |
| `Genetic Risk`                | int64   | Tingkat risiko genetik pasien                                                |
| `chronic Lung Disease`               | int64   | Tingkat penyakit paru-paru kronis pasien                  |
| `Balanced Diet`               | int64   | Tingkat diet seimbang pasien                                        |
| `Obesity`                | int64   | Tingkat obesitas pasien                                     |
| `Smoking`            | int64   | Tingkat merokok pasien                                          |
| `Passive Smoker`     | int64   | Tingkat perokok pasif pasien                            |
| `Chest Pain`                | int64   | Tingkat nyeri dada pasien                           |
| `Coughing of Blood`         | int64   | Tingkat batuk darah pasien                                    |
| `Fatigue`                  | int64   | Tingkat kelelahan pasien                          |
| `Weight Loss`           | int64   | Tingkat penurunan berat badan pasien                                  |
| `Shortness of Breath`        | int64   | Tingkat sesak napas pasien                                         |
| `Wheezing`      | int64   | Tingkat mengi pasien                      |
| `Swallowing Difficulty`     | int64   | Tingkat kesulitan menelan pasien                                   |
| `Clubbing of Finger Nails`         | int64   | Tingkat menggigiti kuku jari pasien                               |
| `Snoring`        | int64   | Frekuensi seseorang mendengkur saat tidur                                         |
| `Dry Cough`      | int64   | Menunjukkan keberadaan batuk kering tanpa dahak                      |
| `Frequent Cold`     | int64   | Seberapa sering pasien mengalami gejala seperti flu, pilek, atau infeksi saluran pernapasan  |
| `Level `                 | object    | Tingkat keparahan atau stadium kondisi kesehatan pasien                                |

### Menangani Missing Value dan Drop Kolom yang tidak digunakan
Pada tahap ini dilakukan pengecekan terhadap data yang tidak valid dalam dataset. Setelah diperiksa, tidak ditemukan adanya nilai null pada kolom manapun. Selanjutnya, kolom 'index' dan 'Patient Id' dihapus karena tidak memiliki kontribusi langsung terhadap analisis

### Menangani Missing Outliers
Outliers merupakan sampel yang nilainya sangat jauh dari cakupan umum data utama dan hasil pengamatan yang kemunculannya sangat jarang dan berbeda dari data hasil pengamatan lainnya.


![Outliers Proyek1](https://github.com/user-attachments/assets/8bdda8be-a138-446d-aff7-a5451cb5e0ae)


Pada bagian ini, terdapat 1 outlier yang terdeteksi, yaitu pada fitur Age. Oleh karena itu, dilakukan penanganan terhadap outlier tersebut, agar tidak memengaruhi hasil analisis dan pemodelan yang akan dilakukan pada tahap selanjutnya.


### EDA - Univariate Analysis
Proses univariate data analysis pada masing-masing fitur numerik.
![Univariate Analysis](https://github.com/user-attachments/assets/a479aa4f-043d-4dc6-ad05-eda78428f667)

Dari hasil visualisasi di atas dapat disimpulkan bahwa:
**Age:** Sebagian besar pasien berusia antara 30 hingga 45 tahun.
**Gender:** Mayoritas pasien berjenis kelamin laki-laki (1).
**Air Pollution:** Banyak pasien memiliki skor paparan polusi udara pada level 6.
**Alcohol use:** Skor konsumsi alkohol didominasi pada level 2 dan 7.
**Dust Allergy:** Skor alergi debu paling banyak muncul pada level 7.
**Occupational Hazards:** Sebagian besar pasien berada pada level 2 risiko pekerjaan.
**Genetic Risk:** Tingkat risiko genetik terbanyak pada level 2.
**Chronic Lung Disease:** Skor tertinggi berada pada level 2.
**Balanced Diet:** Mayoritas pasien memiliki skor pola makan seimbang pada level 7.
**Obesity:** Tingkat obesitas tertinggi berada pada level 2.
**Smoking:** Sebagian besar pasien berada di level 2 dalam hal kebiasaan merokok.
**Passive Smoker:** Terbanyak pada level 2 dan 7.
**Chest Pain:** Skor nyeri dada paling sering muncul di level 2.
**Coughing of Blood:** Skor batuk berdarah terbanyak di level 2 dan 4.
**Fatigue:** Tingkat kelelahan paling tinggi di level 2.
**Weight Loss:** Banyak pasien mengalami penurunan berat badan pada level 2.
**Shortness of Breath:** Skor sesak napas terbanyak di level 2 dan 4.
**Wheezing:** Sebagian besar pasien berada pada level 2.
**Swallowing Difficulty:** Tingkat kesulitan menelan paling sering di level 2.
**Clubbing of Finger Nails:** Skor paling banyak muncul pada level 2.
**Frequent Cold:** Mayoritas pasien mengalami flu sering pada level 2 dan 4.
**Dry Cough:** Sebagian besar pasien berada di level 2 dan 4.
**Snoring:** Skor kebiasaan mendengkur terbanyak di level 2.


### EDA - Multivariate Analysis
Proses multivariate data analysis pada masing-masing fitur numerik.
![Multi](https://github.com/user-attachments/assets/7a0edf50-fb7e-485d-b5f2-30a10200e823)


 * Terdapat korelasi kuat antar beberapa gejala, seperti sesak napas, batuk darah, dan nyeri dada.
 * Pola makan seimbang cenderung berhubungan negatif dengan gejala berat.
 * Beberapa variabel risiko (misalnya polusi, merokok) menunjukkan hubungan dengan gejala klinis.
 * Terlihat adanya kelompok data yang membentuk pola atau klaster.
 * Umur dan gender tidak menunjukkan pengaruh signifikan terhadap gejala secara langsung.
 * Gejala-gejala pernapasan sering muncul bersamaan.
 * Data menunjukkan potensi untuk segmentasi pasien berdasarkan kombinasi faktor risiko dan gejala.

### Correlation Matrix
Memeriksa korelasi atau hubungan antar fitur numerik menggunakan heatmap correlation matrix.


Berdasarkan diagram heatmap di atas, disimpulkan bahwa:
 * Rentang nilai korelasi berada antara -0.3 hingga 1.0.
 * Nilai mendekati 1 menunjukkan korelasi positif yang sangat kuat antar fitur.
 * Nilai mendekati 0 berarti korelasi lemah atau tidak ada korelasi.
 * Nilai negatif mendekati -1 menunjukkan korelasi negatif yang kuat (semakin satu naik, yang lain turun).



## Data Preparation
Pada tahap ini, dilakukan proses data preparation untuk memastikan data siap digunakan dalam model Machine Learning yang akan  dibangun
Pada tahap ini, dilakukan beberapa langkah penting untuk mempersiapkan data sebelum digunakan dalam model Machine Learning. Berikut adalah tahapan yang dilakukan:

1. Pemisahan Fitur dan Label
Dataset dibagi menjadi dua bagian utama:
   * Fitur (X): seluruh kolom input kecuali target.
   * Label (y): kolom Level dijadikan sebagai target/label yang ingin diprediksi.

2. Split Data Latih dan Uji
Dataset dibagi menjadi:
   * Data latih (Training Set): 80%
   * Data uji (Test Set): 20%
     
Teknik ini dilakukan dengan fungsi train_test_split dari sklearn.model_selection.

## Modeling
Setelah proses data preparation selesai, langkah selanjutnya adalah membangun model Machine Learning. Dua algoritma yang digunakan dalam proyek ini adalah Naive Bayes dan K-Nearest Neighbors (KNN). Pemilihan kedua model ini didasarkan pada perbedaan pendekatan algoritmik, sehingga dapat dibandingkan performanya:
1. Naive Bayes
  * Algoritma yang digunakan: GaussianNB dari pustaka sklearn.naive_bayes.
  * Prinsip kerja: Berdasarkan Teorema Bayes dengan asumsi bahwa setiap fitur bersifat independen satu sama lain.
  * Parameter yang digunakan:
        * Tidak dilakukan perubahan parameter dari nilai default.
        * Model secara otomatis mengasumsikan bahwa setiap fitur memiliki distribusi Gaussian (normal).
  * Implementasi:
    
![image](https://github.com/user-attachments/assets/79741df3-3964-4737-848a-b6850d359591)


2. K-Nearest Neighbors (KNN)
  * Algoritma yang digunakan: KNeighborsClassifier dari pustaka sklearn.neighbors.
  * Prinsip kerja: Mengklasifikasikan data berdasarkan mayoritas dari K tetangga terdekat berdasarkan jarak (default: Euclidean).
  * Parameter yang digunakan:
      * n_neighbors = 5 → jumlah tetangga terdekat yang dipertimbangkan.
      * Parameter lain menggunakan nilai default
  * Implementasi:
    
![image](https://github.com/user-attachments/assets/23907d3e-c417-4555-afd0-cf57bef6a58e)


## Evaluation
Tahap evaluasi bertujuan untuk menilai seberapa efektif model Naive Bayes dan K-Nearest Neighbors (KNN) dalam memprediksi risiko kanker paru-paru (Low, Medium, High) berdasarkan data pasien. Evaluasi ini dilakukan menggunakan metrik klasifikasi yang umum: Accuracy, Precision, Recall, F1-Score, serta Confusion Matrix.

### Metrik Evaluasi
#### Beberapa metrik yang digunakan:
   * Accuracy: Seberapa banyak prediksi model yang benar secara keseluruhan.
   * Precision: Seberapa tepat model dalam memprediksi kelas tertentu (khususnya penting untuk kategori High risk).
   * Recall: Kemampuan model untuk menemukan semua kasus yang benar-benar termasuk dalam suatu kelas.
   * F1-Score: Rata-rata harmonik dari precision dan recall; berguna ketika data tidak seimbang.
   * Confusion Matrix: Visualisasi performa klasifikasi per kelas.
     
#### Hasil Evaluasi Model

1. **Naive Bayes**
      * **Accuracy** (88%)
      * **Precision, Recall, dan F1-Score**
  
|                  | Precission  | Recall | F1-Score                                                                 |
|------------------------|----------- |----------- |---------------------------------------------------------------------------|
| `High`                  |  0.82    |  0.92    |   0.87                                                 |
| `Low`                  |  1.00    |  0.93    |   0.97                                                 |
| `Medium`                  |  0.85    |  0.76    |   0.80                                                 |

  * **Confusion Matrix**

![image](https://github.com/user-attachments/assets/3936d22a-930d-44cd-883e-a42240747337)

#### Hasil  Analisa
   * Naive Bayes cukup baik dalam mengklasifikasikan risiko Low dan High.
   * Namun performa pada kelas Medium kurang optimal (recall = 0.76), menunjukkan bahwa model sering gagal mendeteksi pasien dengan risiko Medium.
   * Dalam konteks Business Understanding, ini berarti masih ada potensi pasien berisiko Medium yang tidak teridentifikasi, yang bisa berdampak pada pencegahan dini.

2. **K-Nearest Neighbors (KNN)**
      *  **Accuracy** (100%)
      *  **Precision, Recall, dan F1-Score**
  
|                  | Precission  | Recall | F1-Score                                                                 |
|------------------------|----------- |----------- |---------------------------------------------------------------------------|
| `High`                  |  1.00    |  1.00    |   1.00                                                 |
| `Low`                  |  1.00    |  1.00    |   1.00                                                 |
| `Medium`                  |  1.00    |  1.00    |   1.00                                                 |

  * **Confusion Matrix**
  
![image](https://github.com/user-attachments/assets/fa5dbb7a-bc4f-46e1-8d11-8192606b75e3)

#### Hasil  Analisa
   * KNN menunjukkan performa sempurna pada dataset ini.
   * Hal ini menjawab dengan tepat semua Problem Statements:
       * Memprediksi risiko kanker paru-paru secara akurat.
       * Menentukan bahwa KNN lebih efektif daripada Naive Bayes untuk dataset ini.
       * Evaluasi dengan metrik menunjukkan hasil sempurna.
   * Dari segi Business Goals, KNN memberikan hasil yang sangat meyakinkan untuk deteksi awal pasien dengan risiko kanker paru-paru.



**Perbandingan Akurasi dan Implikasi Bisnis**

| Model         | Accuracy | Kelebihan                                                        | Kekurangan                                                      |
|---------------|----------|------------------------------------------------------------------|-----------------------------------------------|
| Naive Bayes   | 88%      | - Proses training cepat                                          | - Performa kurang pada kelas `Medium` (Recall rendah) |
|               |          | - Akurat pada kelas `Low` dan `High`                             | - Asumsi independensi antar fitur bisa tidak realistis |
| KNN           | 100%     | - Akurasi sempurna di seluruh kelas (`High`, `Medium`, `Low`)    | - Potensi overfitting jika dataset kecil      |
|               |          | - Sederhana dan efektif untuk prediksi berbasis kedekatan data   | - Lebih lambat saat prediksi jika jumlah data besar |


#### Kesimpulan

Model Machine Learning yang dibangun berhasil memenuhi tujuan dalam memprediksi risiko kanker paru-paru berdasarkan data pasien. Dengan akurasi tinggi dan nilai precision, recall, serta F1-score yang sangat baik terutama pada model KNN, model ini menunjukkan performa yang sangat kuat dalam mengklasifikasikan risiko pasien ke dalam kategori Low, Medium, dan High.

Model KNN terbukti sangat efektif dalam menangkap pola dari data pasien dan memberikan hasil klasifikasi yang akurat untuk ketiga kategori risiko. Dengan demikian, model ini sangat potensial untuk digunakan dalam mendukung sistem deteksi dini risiko kanker paru-paru serta membantu tenaga medis dalam proses pengambilan keputusan berbasis data.

## Reference
[[1]](http://ijcs.net/ijcs/index.php/ijcs/article/view/3267)
[[2]](http://ijcs.net/ijcs/index.php/ijcs/article/view/3227/178)
[[3]](https://course-net.com/blog/metode-naive-bayes/8)
[[4]](https://www.journal.sekawan-org.id/index.php/jtim/article/view/178/130)
[[5]](https://www.appliedaicourse.com/blog/knn-algorithm-in-machine-learning/)
