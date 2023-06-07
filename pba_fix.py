from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd 
import numpy as np
import regex as re
import json
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle5 as pickle 
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sentimen Analysis",
    page_icon='https://cdn-icons-png.flaticon.com/512/1998/1998676.png',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an extremely cool app!"
    }
)
st.write("""
<center><h2 style = "text-align: justify;">Aplikasi Analisis Sentimen Pendapat orang tua terhadap pembelajaran daring pada masa Covid-19 dengan algoritma naive bayes</h2></center>
""",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"><img src="https://cdn-icons-png.flaticon.com/512/1998/1998676.png" width="120" height="120"></h3>""",unsafe_allow_html=True), 
        ["Home","Dataset", "Preprocessing", "TFIDF", "Modeling", "Implementation", "Tentang Kami"], 
            icons=['house', 'bar-chart', 'gear', 'list-task','arrow-down-square', 'check2-square', 'person'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#412a7a"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#412a7a"}
            }
        )

    if selected == "Home":
        st.write("""<h3 style = "text-align: center;">
        <img src="https://onlinelearning.binus.ac.id/files/2021/10/Web-Banner-Artikel-BOL-31-06.jpg" width="650" height="300">
        </h3>""",unsafe_allow_html=True)
        st.write(""" <p style = "text-align: justify;">Pandemi COVID-19 merupakan penyakit yang skala penyebarannya terjadi secara global di seluruh dunia, termasuk Indonesia. Banyak bidang yang terkena dampak pandemi ini termasuk pendidikan. Indonesia saat ini sedang menjalankan strategi pembelajaran daring yang menimbulkan banyak opini masyarakat. Analisis sentiment pada cabang Text Mining digunakan untuk mengklasifikasi suatu entitas pada dokumen teks yang terdiri dari dua kelas yaitu positif dan negatif, kelas tersebut diperoleh dengan mengklasifikasikan dataset headline dan substansi berita terkait pembelajaran daring. Tujuan dari aplikasi ini adalah untuk melakukan prediksi pendapat orangtua terhadap pembelajaran daring serta mengetahui nilai akurasi dari pendapat tersebut dengan algoritma Naïve Bayes Classifier</p>""",unsafe_allow_html=True)
        

    elif selected == "Dataset":
        st.write("#### Deskripsi Dataset")
        st.write(""" <p style = "text-align: justify;">Dataset ini diambil dari hasil crawling data berita yang diambli melalui beberapa paltform berita online. Selanjutnya data berita tersebut akan diklasifikasikan ke dalam dua kategori sentimen yaitu negatif dan positif kemudian dilakukan penerapan algoritma Multinomial Naive Bayes untuk mengetahui nilai akurasinya.</p>""",unsafe_allow_html=True)
        st.write("#### Preprocessing Dataset")
        st.write(""" <p style = "text-align: justify;">Preprocessing data merupakan proses dalam mengganti teks tidak teratur supaya teratur yang nantinya dapat membantu pada proses pengolahan data.</p>""",unsafe_allow_html=True)
        st.write(""" 
        <ol>
            <li>Case folding merupakan tahap untuk mengganti keseluruhan kata kapital pada dataset agar berubah menjadi tidak kapital.</li>
            <li>Cleansing yaitu merupakan proses untuk menghilangkan semua simbol, angka, ataupun emoticon yang terdapat didalam dataset</li>
            <li>Slangword Removing yaitu satu proses yang dilakukan untuk mendeteksi dan menghilangkan kata-kata yang tidak baku di dalam dataset</li>
            <li>Steaming yaitu proses yang digunakan untuk menghilangkan semua kata imbuhan dan merubahnya menjadi kata dasar.</li>
        </ol> 
        """,unsafe_allow_html=True)
        st.write("#### Dataset")
        df = pd.read_csv("DataBerita.csv")
        st.write(df)
     
    elif selected == "Preprocessing":
        st.write("## Preprocessing")
        st.subheader("Case Folding")
        case_folding = pd.read_csv(preplowecase.csv")
        st.write(pd.DataFrame(case_folding['Judul Berita'].values,columns=["Case Folding"]))
        st.subheader("Cleansing Dataset")
        cleansing = pd.read_csv(prepcleansing.csv")
        st.write(pd.DataFrame(cleansing['Cleansing Abstrak'].values,columns=["Cleansing Data"]))
        st.subheader("Slang Word")
        slang = pd.read_csv(data_slang.csv")
        st.write(pd.DataFrame(slang['Slang Word Corection'].values,columns=["Slang Word Corection"]))
        st.subheader("Steaming")
        steaming = pd.read_csv(data_steaming.csv")
        st.write(pd.DataFrame(steaming['Steaming'].values,columns=["Steaming"]))
                               
    elif selected == "TFIDF":
        st.subheader("Hasil dari pembobotan kata menggunakan metode TFIDF")
        #Dataset
        Data_ulasan = pd.read_csv(datapba_prep.csv")
        ulasan_dataset = Data_ulasan['Steaming']
        sentimen = Data_ulasan['Label']
                                  
        # TfidfVectorizer 
        with open('tfidf.pkl', 'rb') as file:
            loaded_data_tfid = pickle.load(file)
        tfidf_wm = loaded_data_tfid.fit_transform(ulasan_dataset)
        tfidf_tokens = loaded_data_tfid.get_feature_names_out()
        df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
        st.write(df_tfidfvect)
    
    elif selected == "Modeling":
        st.subheader("Modeling Dataset dengan Metode MultinomialNB")
        #Dataset
        Data_ulasan = pd.read_csv(datapba_prep.csv")
        ulasan_dataset = Data_ulasan['Steaming']
        sentimen = Data_ulasan['Label']

        # TfidfVectorizer 
        with open('tfidf.pkl', 'rb') as file:
            loaded_data_tfid = pickle.load(file)
        tfidf_wm = loaded_data_tfid.fit_transform(ulasan_dataset)
        tfidf_tokens = loaded_data_tfid.get_feature_names_out()
        df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)

        #Train test split
        training, test = train_test_split(tfidf_wm,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
        training_label, test_label = train_test_split(sentimen, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing    

        clf = MultinomialNB(alpha = 0.01)
        st.info(clf.fit(training, training_label))
        y_pred = clf.predict(test)

        st.write("#### Confussion Matrix")
        st.write("""<h3 style = "text-align: left;">
        <img src="data:image/png;base64,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" width="400" height="250">
        </h3>""",unsafe_allow_html=True)

        st.write("#### Classification Report")
        st.write("""<h3 style = "text-align: left;">
        <img src="https://snipboard.io/hoG6s8.jpg" width="400" height="200">
        </h3>""",unsafe_allow_html=True)
        
    elif selected == "Implementation":
        #Getting input from user
        word = st.text_area('Masukkan kata yang akan di analisa :')

        submit = st.button("submit")

        if submit:
            def prep_input_data(word, slang_dict):
                #Lowercase data
                lower_case_isi = word.lower()

                #Cleansing dataset
                clean_symbols = re.sub("[^a-zA-Zï ]+"," ", lower_case_isi)

                #Slang word removing
                def replace_slang_words(text):
                    words = nltk.word_tokenize(text.lower())
                    words_filtered = [word for word in words if word not in stopwords.words('indonesian')]
                    for i in range(len(words_filtered)):
                        if words_filtered[i] in slang_dict:
                            words_filtered[i] = slang_dict[words_filtered[i]]
                    return ' '.join(words_filtered)
                slang = replace_slang_words(clean_symbols)

                #Steaming Data
                factory = StemmerFactory()
                stemmer = factory.create_stemmer()
                stem = stemmer.stem(slang)
                return lower_case_isi,clean_symbols,slang,stem
            
            #Kamus
            with open('combined_slang_words.txt') as f:
                data = f.read()
            slang_dict = json.loads(data)

            #Dataset
            Data_ulasan = pd.read_csv("datapba_prep.csv")
            ulasan_dataset = Data_ulasan['Steaming']
            sentimen = Data_ulasan['Label']

            # TfidfVectorizer 
            with open('tfidf.pkl', 'rb') as file:
                loaded_data_tfid = pickle.load(file)
            tfidf_wm = loaded_data_tfid.fit_transform(ulasan_dataset)

            #Train test split
            training, test = train_test_split(tfidf_wm,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
            training_label, test_label = train_test_split(sentimen, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing    

            # model
            with open('model.pkl', 'rb') as file:
                loaded_model = pickle.load(file)
            clf = loaded_model.fit(training,training_label)
            y_pred=clf.predict(test)

            #Evaluasi
            akurasi = accuracy_score(test_label, y_pred)

            # Inputan 
            lower_case_isi,clean_symbols,slang,stem = prep_input_data(word, slang_dict)
            
            # #Prediksi
            v_data = loaded_data_tfid.transform([stem]).toarray()
            y_preds = clf.predict(v_data)

            st.subheader('Preprocessing')
            st.write(pd.DataFrame([lower_case_isi],columns=["Case Folding"]))
            st.write(pd.DataFrame([clean_symbols],columns=["Cleansing"]))
            st.write(pd.DataFrame([slang],columns=["Slang Word Removing"]))
            st.write(pd.DataFrame([stem],columns=["Steaming"]))

            st.subheader('Akurasi')
            st.info(akurasi)

            st.subheader('Prediksi')
            if y_preds == "positif":
                st.success('Positive')
            else:
                st.error('Negative')

    elif selected == "Tentang Kami":
        st.write("##### Mata Kuliah = Pembelajaran Bahasa Alami - A") 
        st.write('##### Kelompok 1')
        st.write("1. Muhammad Hanif Santoso (200411100078)")
        st.write("2. Alfito Wahyu Kamaly (200411100079)")
        st.write("3. Fajrul Ihsan Kamil (200411100172)")
