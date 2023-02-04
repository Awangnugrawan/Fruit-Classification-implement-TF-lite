# Laporan Proyek Machine Learning – Awang Mulya Nugrawan

## Domain Proyek
Masalah latar belakang dari prediksi tarif pesawat adalah bagaimana memprediksi harga tiket pesawat yang akan dibeli oleh penumpang. Hal ini menjadi masalah karena harga tiket pesawat sangat bervariasi dan dipengaruhi oleh berbagai faktor seperti jenis penerbangan, waktu penerbangan, jarak, musim, dan banyak lagi. Oleh karena itu, sangat penting bagi perusahaan penerbangan untuk memprediksi harga tiket secara akurat agar dapat membuat keputusan strategis dan memaksimalkan keuntungan. Prediksi yang akurat juga membantu dalam memenuhi kebutuhan dan ekspektasi pelanggan dengan menawarkan harga yang wajar dan kompetitif.

**Rubrik/Kriteria Tambahan (Opsional)**:
Sebagian besar penelitian tentang prediksi harga tiket pesawat berfokus pada tingkat nasional atau pasar tertentu. Penelitian pada tingkat segmen pasar, bagaimanapun, masih sangat terbatas.penelitian yang ada pada segmen pasar.Prediksi harga segmen pasar menggunakan model statistik konvensional berbasis heuristik  konvensional, seperti regresi linier dan didasarkan pada asumsi bahwa ada hubungan linier antara variabel dependen dan independen, yang dalam banyak yang dalam banyak kasus, mungkin tidak benar.
Kemajuan terbaru dalam Kecerdasan Buatan (AI) dan Pembelajaran Mesin (ML) memungkinkan untuk menyimpulkan aturan dan variasi model pada harga tiket pesawat berdasarkan sejumlah besar fitur, sering kali mengungkap hubungan tersembunyi di antara fitur-fitur tersebut secara otomatis.
- [ A Framework for Airfare Price Prediction: A Machine Learning Approach] [https://ieeexplore.ieee.org/abstract/document/8843464/]


## Business Understanding

### Problem Statements
Masalah yang sering terjadi adalah karena harga tiket pesawat sangat bervariasi dan dipengaruhi oleh berbagai faktor seperti jenis penerbangan, waktu penerbangan, jarak, musim, dan banyak lagi. Oleh karena itu, sangat penting bagi perusahaan penerbangan untuk memprediksi harga tiket secara akurat agar dapat membuat keputusan strategis dan memaksimalkan keuntungan. Prediksi yang akurat juga membantu dalam memenuhi kebutuhan dan ekspektasi pelanggan dengan menawarkan harga yang wajar dan kompetitif.

Menjelaskan pernyataan masalah latar belakang:
1. Masalah yang pertama terletak pada sulitnya mendapatkan akses ke data, sehingga mereproduksi hasil dan memperluas pekerjaan hampir tidak mungkin dilakukan.
2. Masalah yang kedua adalah bahwa catatan transaksi dari setiap situs pemesanan online adalah sebagian kecil dari total penjualan tiket dari seluruh pasar, membuat data yang diperoleh cenderung miring, dan dengan demikian, tidak mewakili sifat sebenarnya dari seluruh pasar.
3. Fluctuasi harga tiket pesawat: Harga tiket pesawat sering kali berubah-ubah, dan sulit untuk diprediksi dengan tepat.
4. Kebutuhan pelanggan untuk memprediksi harga tiket: Pelanggan sering membutuhkan informasi tentang harga tiket untuk membuat keputusan pembelian yang tepat.
5. Keinginan untuk memaksimalkan profit: Maskapai penerbangan ingin memaksimalkan profit dengan menjual tiket pada harga yang tepat.

### Goals
Menjelaskan tujuan dari pernyataan masalah:
1. Menentukan harga tiket: 
2. Meningkatkan efisiensi: 
3. Membuat keputusan strategis
4. Meningkatkan kinerja bisnis: 
5. Meningkatkan kepuasan pelanggan: 

**Rubrik/Kriteria Tambahan (Opsional)**:
 ### Solution statements
1. Menentukan harga tiket: Prediksi tarif pesawat membantu dalam menentukan harga tiket yang tepat. Ini memastikan bahwa maskapai penerbangan memperoleh keuntungan maksimal dan pelanggan tidak dikenakan biaya yang tidak wajar.
2. Meningkatkan efisiensi: Prediksi tarif pesawat membantu dalam meningkatkan efisiensi bisnis. Dengan memperkirakan permintaan dan harga tiket, maskapai penerbangan dapat mengoptimalkan jumlah kursi yang tersedia dan memastikan bahwa mereka tidak kehilangan peluang bisnis.
3. Membuat keputusan strategis: Prediksi tarif pesawat membantu maskapai penerbangan dalam membuat keputusan strategis, seperti menentukan rute baru, menentukan kapasitas kursi, dan menentukan harga tiket.
4. Meningkatkan kinerja bisnis: Prediksi tarif pesawat membantu dalam meningkatkan kinerja bisnis maskapai penerbangan. Ini memastikan bahwa maskapai penerbangan memperoleh keuntungan yang optimal dan dapat bersaing dengan maskapai penerbangan lain.
5. Meningkatkan kepuasan pelanggan: Prediksi tarif pesawat membantu dalam meningkatkan kepuasan pelanggan. Ini memastikan bahwa pelanggan memperoleh harga tiket yang adil dan membantu dalam membuat pengalaman penerbangan yang lebih baik.

Dari beberapa solusi statement tersebut digunakan 3 algoritma untuk memaksimalkan prediksi harga tiket pesawat dan menggunakan matriks evaluasi R2_Score,MSE dan MAE



## Data Understanding
Dataset yang digunakan pada laporan ini adalah data tarif penerbangan di negara India pada tahun 2019 .Tujuan atau target dari dataset ini adalah menganalisis data  dan membangun model prediksi yang dapat memprediksi harga tiket pesawat berdasarkan fitur-fitur tersebut. [Kaggle Repository] (https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh/code)

### Variabel-variabel pada flight fare prediction Kaggle Dataset adalah sebagai berikut:
1. Airline: kolom ini akan berisi semua jenis maskapai penerbangan seperti Indigo, Jet Airways, Air India, dan masih banyak lagi. 
2. Date_of_Journey: Kolom ini tentang tanggal di mana perjalanan penumpang akan dimulai. 
3. Source: Kolom ini berisi nama tempat dari mana perjalanan penumpang akan dimulai. 
4. Destination: Kolom ini menampung nama tempat tujuan perjalanan penumpang. 
5. Route: Kolom ini tentang rute apa yang dipilih penumpang untuk melakukan perjalanan dari tempat asal ke tempat tujuan. 
6. Dep_time : merujuk pada waktu keberangkatan (departure time) suatu penerbangan,
7. Arrival_Time: Waktu kedatangan adalah kapan penumpang akan sampai di tempat tujuan. 
8. Duration: Durasi adalah seluruh periode yang dibutuhkan penerbangan untuk menyelesaikan perjalanannya dari sumber ke tujuan. 
9. Total_Stops: Kolom ini tentang berapa banyak tempat penerbangan akan berhenti di sana untuk penerbangan sepanjang perjalanan. 
10. Additional_Info: Pada kolom ini, kita akan mendapatkan informasi tentang makanan, jenis makanan, dan fasilitas lainnya. 
11. Price: Harga penerbangan untuk perjalanan lengkap termasuk semua biaya sebelum naik pesawat.
 
**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.
- Explorasi data analysis:
- Jumlah baris:10683
- Jumlah kolom : 11
- Tipe data tiap feature:
1.	Airline  			=  object
2.	Date_of_Journey  		=  object
3.	 Source   			=  object
4.	Destination      		=  objecth
5.	 Route            			=   object
6.	Dep_Time         		=  object
7.	Arrival_Time     		=  object
8.	 Duration         			=  object
9.	Total_Stops      		=  object
10.	 Additional_Info  		= object
11.	Price            			=  int64

-	Nilai null = 
Airline            		0
Date_of_Journey         0
Source             	0
Destination        	0
Route              		1
Dep_Time           	0
Arrival_Time       	0
Duration           	0
Total_Stops        	1
Additional_Info    	0
Price              		0

Dari hasil visualisasi dengan menggunakan diagram batang pada atribut “Airline” dapat dilihat bahwa jenis maskapai penerbangan  yang paling banyak adalah Jet airways disusul dengan IndiGo kemudian AirIndia sedangkan maskapai penerbangan Visitera premium economy adalah maskapai yang paling sedikit.



## Data Preparation
Teknik yang digunakan pada notebook secara berurutan.
-	Missing value
-	Handling "Date_of_Journey"
-	Handling "Dep_Time"
-	Handling "Arrival_Time"
-	Handling "Duration"
-	Handling "Total_stops"
-	Handling "Airplane"
-	Handling "Source"
-	Handling "Destination"
-	Concenate dataframe Airline,Source, dan Destination


**Rubrik/Kriteria Tambahan (Opsional)**: 
-	Missing value
 Teknik yang pertama dilakukan adalah dengan mengecek nilai null pada dataset setelah menggunakan code “ df.isnull().sum()” di dapatkan bahwa terdapat 2 nilai null pada masing masing kolom route dan top_stops. Setelah itu dilakukan drop atribut yang tidak di perlukan seperti "Route" karena valuenya mirip dengan kolom Total Stops dan "Additional info" karena sebagian besar valuenya adalah no info.

-	Handling atribut “Date_of_Journey”:
Selanjutnya adalah penanganan pada atribut tentang tanggal keberangkatan dari maskapai penerbangan. Value nya adalah dalam format dd/mm/yy , untuk mempermudah dalam modelling maka kita akan memisahkan nya menjadi tiap kolom menjadi kolom hari “Journey_days” dan kolom bulan “Journey_month” sedangkan untuk tahun tidak perlu karena semua value pada dataset ini sama yaitu tahun 2019. Dan setelah proses tersebut dilakukan maka kolom asal “Date_of_Journey” dapat di hapus/drop.

-	Handling atribut “Dep_Time”:
Pada proses ini mirip dengan proses sebelumnya kita akan memisahkan value nya menjadi kolom menit “Dep_minute” dan kolom jam “Dep_Time”. Selanjutnya kita dapat melakukan drop pada kolom aslinya “Dep_Time”

-	Handling atribut “Arrival_time”:
Atribut ini tentang waktu keberangkatan juga dapat dipisahkan menjadi waktu keberangkatan dalam menit “Arrival_Minute” dan keberangkatan dalam jam “Arrival_hour”. Kemudian kolom asalnya di hapus

-	Handling atribut “Duration”:
Sama seperti proses sebelumnya pada kolom ini kita juga akan memisahkan nya menjadi durasi dalam menit “Duration_mins” dan Durasi dalam jam “Duration_hour”. Seperti biasa kolom asalnya dapat di drop

-	Handling atribut “Total_stops”:
Pada atribut ini terdapat 5 value utama yang dapat dilakukan label encoding dengan nilai 0-4

-	Handling Atribut “Airplane”:
Berbeda dengan atribut-atribuk numerik sebelumnya, pada kolom akan dilakukan penyederhanaan atau penggabungan value. Untuk value yang di gabung adalah value yang memiliki jumlah sedikit seperti 'Trujet','Vistara Premium economy','Jet Airways Business','Multiple carriers Premium economy' maka akan di gabung menjadi satu value yaitu “Other”. Setelah itu kita simpan dalam variabel dataframe Airline dan melakukan OneHotEncoding pada tiap variabel

-	Handling atribut “Source”:
Pada atribut ini dapat kita simpan dalam variabel dataframe Source dan melakukan OneHotEncoding pada tiap variabel

-	Handling atribut “Destination”
Sama seperti atribut Airline, kita juga dapat melakukan penyederhanaan / penggabungan value yang memilki makna yang sama yaitu “New Delhi” dan “Delhi” sehingga dapat di satukan menjadi “Delhi”. Selanjutnya dapat kita simpan dalam variabel dataframe Destination dan melakukan OneHotEncoding pada tiap variabel.

-	Concenate dataframe Airline,Source, dan Destination:
Penyatuan 3 atribut catagorik yang telah dilakukan teknik One hot encoding pada tiap variabelnya ke dataframe utama “df”.



## Modeling
Pada dataset ini menggunakan 3 modelling yaitu :
1.	KNeighborsRegressor menggunakan k = 3 tetangga dan metric Euclidean untuk mengukur jarak antara titik
2.	RandomForestRegressor menggunakan n_estimators=50, max_depth=16, random_state=55, n_jobs=-1
3.	DecisionTreeRegressor menggunakan max_depth=20, random_state=3

**Rubrik/Kriteria Tambahan (Opsional)**:
 
Kelebihan dan kekurangan tiap Algoritma:
1. KNeighborsRegressor 
Kelebihan:
-	Mudah diimplementasikan: KNeighborsRegressor sangat mudah diimplementasikan dan bisa digunakan hanya dengan beberapa baris kode.
-	Menangani data yang hilang: KNeighborsRegressor bisa menangani data yang hilang tanpa memerlukan imputasi apa pun.
-	Berfungsi dengan baik pada dataset kecil: KNeighborsRegressor berfungsi dengan baik pada dataset kecil dan merupakan pilihan yang baik saat data yang tersedia terbatas.
Kekurangan:
-	Sensitif terhadap fitur yang tidak relevan: KNeighborsRegressor sensitif terhadap fitur yang tidak relevan dan bisa terpengaruh oleh adanya fitur bising atau berlebihan dalam data.
-	Mahal secara komputasional: KNeighborsRegressor bisa mahal secara komputasional saat jumlah titik data besar, karena algoritma harus menghitung jarak antara semua titik data.
-	Kinerja tergantung pada pilihan k: Kinerja KNeighborsRegressor tergantung pada pilihan k, jumlah tetangga terdekat untuk dipertimbangkan, yang bisa sulit ditentukan.

Pada dataset nilai K terbaik setelah pengujian beberapa nilai K didapatkan nilai K terbaik adalah 3 .Namun algoritma ini bukan hasil yang terbaik pada ketiga modelling. Selain itu , ini juga merupakan kekurangan karena nilai K yang sulit ditemukan

2. Random Forest Regressor
Kelebihan:
-	Bisa menangani hubungan non-linier: Random Forest Regressor bisa menangani hubungan non-linier antara fitur dan variabel target, membuatnya pilihan yang baik untuk dataset yang kompleks.
-	Tahan terhadap outliers: Random Forest Regressor tahan terhadap outliers dan tidak membuat asumsi yang kuat tentang distribusi data.
-	Mengurangi overfitting: Dengan mengambil rata-rata prediksi dari banyak pohon keputusan, Random Forest Regressor mengurangi overfitting, yang merupakan masalah umum pada algoritma pohon keputusan.
Kekurangan:
-	Mahal secara komputasional: Random Forest Regressor bisa mahal secara komputasional, terutama saat jumlah pohon besar atau saat jumlah fitur tinggi.
-	Rawan overfitting: Meskipun Random Forest Regressor kurang rawan overfitting dibandingkan pohon keputusan, ia masih bisa overfitting jika jumlah pohon terlalu besar atau jika kedalaman pohon terlalu dalam.
-	Sulit diterjemahkan: Berbeda dengan model regresi linier sederhana, prediksi Random Forest Regressor sulit diterjemahkan, karena berdasarkan pada kombinasi dari banyak pohon keputusan.

Pada modelling menggunakan n_estimators=50, max_depth=16, random_state=55, n_jobs=-1 dengan parameter tersebut memberikan hasil algoritma paling baik diantara algoritma lainnya

3. Decision Tree Regressor
Kelebihan 
-	Mudah dipahami dan diimplementasikan: Decision Tree Regressor memiliki representasi visual yang mudah dipahami, sehingga memudahkan interpretasi hasil dan membuat model ini mudah dipahami oleh stakeholder.
-	Dapat menangani fitur numerik dan kategorikal: Decision Tree Regressor dapat menangani fitur numerik dan kategorikal dengan baik, sehingga dapat digunakan untuk berbagai jenis data.
-	Dapat menangani outliers dan non-linearitas: Decision Tree Regressor memiliki kemampuan membagi data secara berulang-ulang sehingga dapat menangani outlier dan non-linearitas dalam data.
Kekurangan :
-	Mudah overfitting: Decision Tree Regressor memiliki kecenderungan untuk overfitting jika depth-nya terlalu dalam. Ini dapat diatasi dengan teknik seperti pemotongan pohon, tetapi membutuhkan pemahaman yang baik dari model dan dataset.
-	Instabilitas: Decision Tree Regressor sangat sensitif terhadap perubahan kecil pada data, sehingga model yang dibangun dengan dataset yang berbeda mungkin sangat berbeda.
-	Bias terhadap fitur yang memiliki banyak data: Decision Tree Regressor cenderung memprioritaskan fitur yang memiliki banyak data dalam membuat pembagian data.

Pada modelling ini menggunakan parameter max_depth=20 , namun setelah di evaluasi terjadi overfitting sesuai dengan kekurangan yang dijelaskan di atas. Sehingga Algoritma ini tidak lebih baik daripada Random Forest Regressor tapi sedikit lebih baik dari Algoritma KNeighborsRegressor


## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.
Pada tahap evaluasi digunakan tiga metrik evaluasi yang digunakan yaitu:
1.	R2_Score
Dengan menggunakan metrik R2_Score tersebut di dapatkan hasil dari 3 modelling :
                train	                   test
KNN	0.798497	0.624805
RF	0.941121	0.825389
DTR	0.972591	0.737278

2.	Mean Square Error
Dengan menggunakan metrik Mean Square Error tersebut di dapatkan hasil dari 3 modelling:
	    train	                    test
KNN	4186.297999	8712.025152
RF	1223.235661	4054.469
DTR	569.42498	6100.403468

3.	Mean Absolute Error
Dengan menggunakan metrik Mean Absolute Error
tersebut di dapatkan hasil dari 3 modelling :
	train	                  test
KNN	1.24639	1.799676
RF	0.690195	1.236153
DTR	0.296922	1.440906

Dari hasil Evaluasi tersebut disimpulkan bahwa model Random Forest Regressor adalah yang terbaik diantara yang lain

**Rubrik/Kriteria Tambahan (Opsional)**: 
1. R2_Score
R2 Score adalah metrik yang digunakan untuk mengukur seberapa baik model regresi memprediksi target. Formula R2 Score adalah:
R2 = 1 - (SSres / SStot)
Ket:
SSres adalah sum of squared residuals, yaitu jumlah kuadrat selisih antara nilai target aktual dan nilai target prediksi.
SStot adalah total sum of squares, yaitu jumlah kuadrat selisih antara nilai target aktual dan nilai rata-rata target.
Metrik ini mengukur seberapa baik model regresi menjelaskan variasi dari target (tarif pesawat). Nilai R2_Score berkisar antara 0 dan 1, dimana nilai 1 menunjukkan model regresi yang sempurna dan nilai 0 menunjukkan model regresi yang buruk. Dalam hal prediksi tarif pesawat, nilai R2_Score yang tinggi menunjukkan bahwa model regresi memiliki kemampuan yang baik dalam memprediksi tarif pesawat.

2. Mean Square Error
Mean Squared Error (MSE) adalah metrik yang digunakan untuk mengukur kualitas model regresi. Formula MSE adalah:
MSE = (1 / n) * Σ (yi - ŷi)^2
ket:
n adalah jumlah data
yi adalah nilai target aktual
ŷi adalah nilai target prediksi
Σ (yi - ŷi)^2 adalah jumlah kuadrat selisih antara nilai target aktual dan nilai target prediksi

Metrik ini mengukur rata-rata kuadrat selisih antara nilai target aktual (tarif pesawat) dan nilai target prediksi. Semakin kecil nilai MSE, semakin baik model regresi dalam memprediksi tarif pesawat. Dalam hal prediksi tarif pesawat, model regresi dengan MSE yang lebih kecil akan dianggap memiliki performa yang lebih baik dibandingkan dengan model yang memiliki MSE yang lebih besar.

3.Mean Absolute Error
Mean Absolute Error (MAE) adalah metrik yang digunakan untuk mengukur kualitas model regresi. Formula MAE adalah:

MAE = (1 / n) * Σ |yi - ŷi|
ket:
n adalah jumlah data
yi adalah nilai target aktual
ŷi adalah nilai target prediksi
Σ |yi - ŷi| adalah jumlah absolute selisih antara nilai target aktual dan nilai target prediksi

Metrik ini mengukur rata-rata selisih antara nilai target aktual (tarif pesawat) dan nilai target prediksi. Semakin kecil nilai MAE, semakin baik model regresi dalam memprediksi tarif pesawat. Dalam hal prediksi tarif pesawat, model regresi dengan MAE yang lebih kecil akan dianggap memiliki performa yang lebih baik dibandingkan dengan model yang memiliki MAE yang lebih besar.


**---Ini adalah bagian akhir laporan---**
