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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
import pickle5 as pickle 
from sklearn.metrics import confusion_matrix, accuracy_score

st.write("##PEMROSESAN BAHASA ALAMI A")
st.write("#### Dosen Pengampu : Dr. FIKA HASTARITA RACHMAN, ST., M.Eng")
st.write("#### Kelompok : 1")
st.write("##### Muhammad Hanif Santoso - 200411100078")
st.write("##### Alfito Wahyu Kamaly - 200411100079")
st.write("##### Fajrul Ihsan Kamil - 200411100172")

#Navbar
implementation = st.tabs(["Implementation"])

st.write("""
<center><h4 style = "text-align: justify;">Aplikasi Analisis Sentimen Pendapat orang tua terhadap pembelajaran daring pada masa Covid-19 dengan algoritma naive bayes</h4></center>
""",unsafe_allow_html=True)

#Fractional Knapsack Problem
#Getting input from user
word = st.text_area('Masukkan kata yang akan di analisa :')

submit = st.button("submit")
if submit:
    def prep_input_data(word, slang_dict):
        lower_case_isi = word.lower()
        clean_symbols = re.sub("[^a-zA-Zï ]+"," ", lower_case_isi)
        def replace_slang_words(text):
            words = nltk.word_tokenize(text.lower())
            words_filtered = [word for word in words if word not in stopwords.words('indonesian')]
            for i in range(len(words_filtered)):
                if words_filtered[i] in slang_dict:
                    words_filtered[i] = slang_dict[words_filtered[i]]
            return ' '.join(words_filtered)
        slang = replace_slang_words(clean_symbols)
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stem = stemmer.stem(slang)
        return lower_case_isi,clean_symbols,slang,stem

    #Kamus
    with open('combined_slang_words.txt') as f:
        data = f.read()
    slang_dict = json.loads(data)

    #Dataset
    df = pd.read_csv('https://raw.githubusercontent.com/HanifSantoso05/dataset_matkul/main/DataBerita.csv')
    dataset_prep = []
    with open('test.txt', 'r') as fp:
        for line in fp:
            x = line[:-1]
            dataset_prep.append(x)

    # TfidfVectorizer 
    with open('model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    with open('tfidf.pkl', 'rb') as file:
        loaded_data_tfidf = pickle.load(file)

    tfidf_wm = loaded_data_tfidf.fit_transform(dataset_prep)

    #Train test split
    training, test = train_test_split(tfidf_wm,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(df['Label'], test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing    

    #model
    clf = loaded_model.fit(training, training_label)
    y_pred = clf.predict(test)

    #Evaluasi
    akurasi = accuracy_score(test_label, y_pred)

    # #Inputan 
    lower_case_isi,clean_symbols,slang,stem = prep_input_data(word, slang_dict)

    #Prediksi
    v_data = loaded_data_tfidf.transform([stem]).toarray()
    y_preds = clf.predict(v_data)

    st.subheader('Preprocessing')
    st.write("Case Folding :",lower_case_isi)
    st.write("Cleansing :",clean_symbols)
    st.write("Slang Word :",slang)
    st.write("Steaming :",stem)

    st.subheader('Akurasi')
    st.info(akurasi)

    st.subheader('Prediksi')
    if y_preds == "Positif":
        st.success('Positive')
    else:
        st.error('Negative')



