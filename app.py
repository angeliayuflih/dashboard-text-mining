# =====================================================
# IMPORT LIBRARY
# =====================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from collections import Counter

from sklearn.feature_extraction.text import (
    CountVectorizer
)

import joblib

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen",
    layout="wide"
)

# =====================================================
# LOAD DATA & MODEL
# =====================================================
data = pd.read_csv("data.csv")

# menghindari error NaN
data['clean_text'] = (
    data['clean_text']
    .fillna('')
    .astype(str)
)

model = joblib.load("model_lr.pkl")
tfidf = joblib.load("tfidf.pkl")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "Home",
        "Dataset",
        "EDA",
        "Modeling",
        "Prediksi Sentimen"
    ]
)
# =====================================================
# HOME
# =====================================================
if menu == "Home":

    st.title("Dashboard Analisis Sentimen")

    st.write("""
    Dashboard analisis sentimen review
    Taman Budaya Embung Giwangan.
    """)

    # metric
    total = len(data)

    positif = len(
        data[data['sentimen'] == 'positif']
    )

    negatif = len(
        data[data['sentimen'] == 'negatif']
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Review", total)
    col2.metric("Positif", positif)
    col3.metric("Negatif", negatif)

    st.subheader("Wordcloud")

    vectorizer = CountVectorizer(
    ngram_range=(2,2)
    )

    X = vectorizer.fit_transform(
    data[
        data['sentimen'] == 'positif'
    ]['clean_text']
    )

    sum_words = X.sum(axis=0)

    words_freq = [
    (
        word,
        sum_words[0, idx]
    )
    for word, idx
    in vectorizer.vocabulary_.items()
    ]

    words_freq = sorted(
    words_freq,
    key=lambda x: x[1],
    reverse=True
    )

    # ubah ke dictionary
    bigram_dict = dict(words_freq)

    # buat wordcloud
    wordcloud_bigram = WordCloud(
     width=1000,
     height=500,
     background_color='white'
    ).generate_from_frequencies(
    bigram_dict
    )

# tampilkan
    fig3, ax6 = plt.subplots(
    figsize=(12,6)
    )

    ax6.imshow(
    wordcloud_bigram,
    interpolation='bilinear'
    )

    ax6.axis('off')

    st.pyplot(fig3)

# =====================================================
# DATASET
# =====================================================
elif menu == "Dataset":

    st.title("Dataset Review")

    st.dataframe(data)

    st.write(
        "Jumlah Data:",
        data.shape[0]
    )

# =====================================================
# EDA
# =====================================================
elif menu == "EDA":

    st.title("Exploratory Data Analysis")

    # =========================================
    # Distribusi Sentimen
    # =========================================
    st.subheader("Distribusi Sentimen")

    fig1, ax1 = plt.subplots()

    data['sentimen'].value_counts().plot(
        kind='bar',
        ax=ax1
    )

    plt.xticks(rotation=0)

    st.pyplot(fig1)

    # =========================================
    # Distribusi Jumlah Kata
    # =========================================
    st.subheader("Distribusi Jumlah Kata")

    data['jumlah_kata'] = (
        data['clean_text']
        .apply(lambda x: len(x.split()))
    )

    fig2, ax2 = plt.subplots()

    ax2.hist(data['jumlah_kata'])

    ax2.set_xlabel("Jumlah Kata")
    ax2.set_ylabel("Frekuensi")

    st.pyplot(fig2)

    # =========================================
    # WORDCLOUD
    # =========================================
    st.subheader("Wordcloud Bigram Positif")

    vectorizer = CountVectorizer(
    ngram_range=(2,2)
    )

    X = vectorizer.fit_transform(
    data[
        data['sentimen'] == 'positif'
    ]['clean_text']
    )

    sum_words = X.sum(axis=0)

    words_freq = [
    (
        word,
        sum_words[0, idx]
    )
    for word, idx
    in vectorizer.vocabulary_.items()
    ]

    words_freq = sorted(
    words_freq,
    key=lambda x: x[1],
    reverse=True
    )

    # ubah ke dictionary
    bigram_dict = dict(words_freq)

    # buat wordcloud
    wordcloud_bigram = WordCloud(
     width=1000,
     height=500,
     background_color='white'
    ).generate_from_frequencies(
    bigram_dict
    )

# tampilkan
    fig3, ax6 = plt.subplots(
    figsize=(12,6)
    )

    ax6.imshow(
    wordcloud_bigram,
    interpolation='bilinear'
    )

    ax6.axis('off')

    st.pyplot(fig3)
    # =========================================
    # TOP WORDS POSITIF
    # =========================================
    st.subheader("Top Words Positif")

    positif_text = ' '.join(
        data[
            data['sentimen'] == 'positif'
        ]['clean_text']
    )

    counter_pos = Counter(
        positif_text.split()
    )

    top_pos = counter_pos.most_common(10)

    df_pos = pd.DataFrame(
        top_pos,
        columns=[
            'Kata',
            'Frekuensi'
        ]
    )

    fig4, ax4 = plt.subplots(
        figsize=(10,5)
    )

    ax4.bar(
        df_pos['Kata'],
        df_pos['Frekuensi']
    )

    plt.xticks(rotation=45)

    st.pyplot(fig4)

    # =========================================
    # TOP BIGRAM POSITIF
    # =========================================
    st.subheader("Top Bigram Positif")

    vectorizer = CountVectorizer(
        ngram_range=(2,2)
    )

    X = vectorizer.fit_transform(
        data[
            data['sentimen'] == 'positif'
        ]['clean_text']
    )

    sum_words = X.sum(axis=0)

    words_freq = [
        (
            word,
            sum_words[0, idx]
        )
        for word, idx
        in vectorizer.vocabulary_.items()
    ]

    words_freq = sorted(
        words_freq,
        key=lambda x: x[1],
        reverse=True
    )

    top_bigram = pd.DataFrame(
        words_freq[:10],
        columns=[
            'Bigram',
            'Frekuensi'
        ]
    )

    fig6, ax6 = plt.subplots(
        figsize=(10,5)
    )

    ax6.barh(
        top_bigram['Bigram'],
        top_bigram['Frekuensi']
    )

    ax6.invert_yaxis()

    st.pyplot(fig6)

    # =========================================
    # TOP BIGRAM POSITIF
    # =========================================
    st.subheader("Top Bigram Negatif")

    vectorizer = CountVectorizer(
        ngram_range=(2,2)
    )

    X = vectorizer.fit_transform(
        data[
            data['sentimen'] == 'negatif'
        ]['clean_text']
    )

    sum_words = X.sum(axis=0)

    words_freq = [
        (
            word,
            sum_words[0, idx]
        )
        for word, idx
        in vectorizer.vocabulary_.items()
    ]

    words_freq = sorted(
        words_freq,
        key=lambda x: x[1],
        reverse=True
    )

    top_bigram = pd.DataFrame(
        words_freq[:10],
        columns=[
            'Bigram',
            'Frekuensi'
        ]
    )

    fig7, ax6 = plt.subplots(
        figsize=(10,5)
    )

    ax6.barh(
        top_bigram['Bigram'],
        top_bigram['Frekuensi']
    )

    ax6.invert_yaxis()

    st.pyplot(fig7)

# =====================================================
# MODELING
# =====================================================
elif menu == "Modeling":

    st.title("Hasil Modeling")

    hasil = pd.DataFrame({

        'Model': [
            'Naive Bayes',
            'LinearSVC',
            'Logistic Regression'
        ],

        'Accuracy': [
            0.91,
            0.88,
            0.87
        ],

        'Recall Negatif': [
            0.00,
            0.36,
            0.73
        ]
    })

    st.dataframe(hasil)

    st.success("""
    Logistic Regression dipilih sebagai
    model terbaik karena memiliki
    recall negatif paling tinggi
    sehingga lebih mampu mengenali
    review negatif.
    """)

# =====================================================
# PREDIKSI SENTIMEN
# =====================================================
elif menu == "Prediksi Sentimen":

    st.title("Prediksi Sentimen")

    user_input = st.text_area(
        "Masukkan Review"
    )

    if st.button("Prediksi"):

        text_tfidf = tfidf.transform(
            [user_input]
        )

        hasil = model.predict(
            text_tfidf
        )

        st.success(
            f"Sentimen: {hasil[0]}"
        )
    