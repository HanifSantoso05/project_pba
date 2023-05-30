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

st.title("PEMROSESAN BAHASA ALAMI A")
st.write("### Dosen Pengampu : Dr. FIKA HASTARITA RACHMAN, ST., M.Eng")
st.write("#### Kelompok : 5")
st.write("##### Hambali Fitrianto - 200411100074")
st.write("##### Pramudya Dwi Febrianto - 200411100042")
st.write("##### Febrian Achmad Syahputra - 200411100106")

#Navbar
implementation = st.tabs(["Implementation"])

with implementation:
    st.write("""
    <center><h4 style = "text-align: justify;">Aplikasi Analisis Sentimen Pendapat orang tua terhadap pembelajaran daring pada masa Covid-19 dengan algoritma naive bayes</h4></center>
    """,unsafe_allow_html=True)

    #Fractional Knapsack Problem
    #Getting input from user
    word = st.text_area('Masukkan kata yang akan di analisa :')

    submit = st.button("submit")

    
