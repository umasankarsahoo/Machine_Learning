# Load libraries
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

url1 = "/Users/umasankarsahoo/Desktop/PyML/Stay_Top_Mobile_Operator_Sales/DS1.csv"
url2 = "/Users/umasankarsahoo/Desktop/PyML/Stay_Top_Mobile_Operator_Sales/DS2.csv"
url3 = "/Users/umasankarsahoo/Desktop/PyML/Stay_Top_Mobile_Operator_Sales/DS3.csv"
url4 = "/Users/umasankarsahoo/Desktop/PyML/Stay_Top_Mobile_Operator_Sales/DS4.csv"
url5 = "/Users/umasankarsahoo/Desktop/PyML/Stay_Top_Mobile_Operator_Sales/DS5.csv"
url6 = "/Users/umasankarsahoo/Desktop/PyML/Stay_Top_Mobile_Operator_Sales/DS6.csv"

# class distribution

df1=pd.read_csv(url1)
df2=pd.read_csv(url2)
df3=pd.read_csv(url3)
df4=pd.read_csv(url4)
df5=pd.read_csv(url5)
df6=pd.read_csv(url6)

#Combining all the data sets 

all_data = pd.concat((
					  df1.loc [:, 'Outlet Size':'Selling price'],
                      df2.loc [:, 'Outlet Size':'Selling price'],
                      df3.loc [:, 'Outlet Size':'Selling price'],
                      df4.loc [:, 'Outlet Size':'Selling price'],
                      df5.loc [:, 'Outlet Size':'Selling price'],
                      df6.loc [:, 'Outlet Size':'Selling price'],
                      )
                    )

# Dropping columns with NaN Values
all_data=all_data.dropna(subset = ['Type_of_pack', 'CITY'])

#Consider the packs corresponding to Data packs

all_data = all_data[(all_data.Type_of_pack == 2) | (all_data.Type_of_pack == 5) | (all_data.Type_of_pack == 7)]


#Numerical to categorical conversion
all_data['Type_of_pack']=['Voucher Elektronik - Paket Internet' if i==2 else 'Kartu Perdana - data' if i==5 else 'Voucher Fisik - Paket Internet' for i in all_data['Type_of_pack']]

all_data['CITY']=[
'Kota Banda Aceh' if i==1
else 'Kota Pematang Siantar'     if i==2
else 'Greater Medan (Medan + Lubuk Pakam)' if i==3
else 'Pekanbaru' if i==4
else 'Batam' if i==5
else 'Padang' if i==6
else 'Kota Jambi' if i==7
else 'Kota Bengkulu' if i==8
else 'Lampung' if i==9
else 'Palembang' if i==10
else 'Babel (Bangka + Belitung)' if i==11
else 'Jakarta Barat' if i==12
else 'jakarta Pusat' if i==13
else 'Jakarta Selatan' if i==14
else 'Jakarta Timur' if i==15
else 'Jakarta Utara' if i==16
else 'Bekasi' if i==17
else 'Kota Depok' if i==18
else 'Kab. Bekasi (kota cikarang)&Kab. Karawang' if i==19
else 'Kab. Pandeglang&Kab. Lebak' if i==20
else 'Kab. Tangerang' if i==21
else 'Tangerang' if i==22
else 'Serang' if i==23
else 'BSKB (Bogor + Sukabumi)'   if i==24
else 'Kab. Indramayu' if i==25
else 'Kab. Bandung' if i==26
else 'Tasikmalaya' if i==27
else 'Bandung' if i==28
else 'Cirebon' if i==29
else 'Purwakarta' if i==30
else 'Yogyakarta' if i==31
else 'Semarang' if i==32
else 'Tegal' if i==33
else 'Pekalongan' if i==34
else 'Solo' if i==35
else 'Kab. Banyumas' if i==36
else 'Kab. Bantul' if i==37
else 'Kota Madiun' if i==38
else 'Kota Malang' if i==39
else 'Kab. Sidoarjo' if i==40
else 'Surabaya' if i==41
else 'Madura' if i==42
else 'Banyuwangi Jember' if i==43
else 'Kab. Buleleng' if i==44
else 'Kab. Gianyar' if i==45
else 'Kab. Lombok Timur' if i==46
else 'Kab. Sumbawa' if i==47
else 'Mataram' if i==48
else 'Denpasar' if i==49
else 'Kota Pontianak' if i==50
else 'Kota Palangkaraya' if i==51
else 'Balikpapan' if i==52
else 'Banjarmasin' if i==53
else 'Makassar' if i==54
else 'Manado' if i==55
else 'Kendari' if i==56 else 'No city' for i in all_data['CITY'] ]

#Plotting Distribution of city wise sales distribution

pd.crosstab(all_data.CITY,all_data.Type_of_pack).plot(kind='bar')
plt.title('City wise sales distribution')
plt.xlabel('City')
plt.ylabel('Frequency of sales')
plt.show()

