import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

# Load and preprocess
df = pd.read_csv('dataout/candidates_news.csv')
df['date_published'] = pd.to_datetime(df['date_published']).dt.tz_localize(None)

# Precompute months
df['month_period'] = df['date_published'].dt.to_period('M')
available_months = sorted(df['month_period'].unique())

# Layout
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.title("Escucha Social - Análisis de Candidatos")

# Tabs
tab1, tab2, tab3 = st.tabs(["Análisis Candidatos", "Comparativo", "Análisis de Narrativa"])

# === TAB 1: Análisis Candidatos ===
with tab1:
    st.header("Análisis de un Candidato")

    candidate = st.selectbox("Selecciona un candidato", df['index'].unique())
    start_month, end_month = st.select_slider(
        "Rango de meses",
        options=available_months,
        value=(available_months[0], available_months[-1]),
        format_func=lambda x: x.strftime("%b %Y")
    )

    start_date = start_month.to_timestamp()
    end_date = end_month.to_timestamp(how='end')

    filtered = df[
        (df['index'] == candidate) &
        (df['date_published'] >= start_date) &
        (df['date_published'] <= end_date)
    ]

    with st.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("Menciones", len(filtered))
        col2.metric("Positivas", (filtered['tono'] == 'positivo').sum())
        col3.metric("Negativas", (filtered['tono'] == 'negativo').sum())

    st.subheader("Serie histórica de menciones")
    mentions_ts = filtered.groupby(filtered['date_published'].dt.to_period('M')).size().reset_index(name='Menciones')
    mentions_ts['date_published'] = mentions_ts['date_published'].dt.to_timestamp()
    fig1 = px.line(mentions_ts, x='date_published', y='Menciones', markers=True)
    fig1.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Distribución de sentimientos")
    sentiment = filtered.groupby([filtered['date_published'].dt.to_period('M'), 'tono']).size().unstack().fillna(0)
    sentiment = sentiment.div(sentiment.sum(axis=1), axis=0) * 100
    sentiment.index = sentiment.index.to_timestamp()
    fig2 = px.area(sentiment)
    fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300)
    st.plotly_chart(fig2, use_container_width=True)

# === TAB 2: Comparativo ===
with tab2:
    st.header("Comparativo entre Candidatos")

    candidate_list = df['index'].unique()

    col1, col2 = st.columns(2)
    with col1:
        cand1 = st.selectbox("Candidato A", candidate_list, key='cand1')
    with col2:
        cand2 = st.selectbox("Candidato B", candidate_list, key='cand2')

    if cand1 == cand2:
        st.warning("Por favor selecciona dos candidatos distintos para el comparativo.")
    else:
        df1 = df[df['index'] == cand1]
        df2 = df[df['index'] == cand2]

        st.subheader("Serie histórica de menciones mensuales")
        m1 = df1.groupby(df1['date_published'].dt.to_period('M')).size().rename(cand1)
        m2 = df2.groupby(df2['date_published'].dt.to_period('M')).size().rename(cand2)
        comparison = pd.concat([m1, m2], axis=1).fillna(0)
        comparison.index = comparison.index.to_timestamp()

        fig3 = px.line(comparison, markers=True)
        fig3.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300)
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Porcentaje de noticias positivas")
        p1 = df1.groupby(df1['date_published'].dt.to_period('M'))['tono'].apply(
            lambda x: (x == 'positivo').sum() / len(x) * 100).rename(f"% Positivas {cand1}")
        p2 = df2.groupby(df2['date_published'].dt.to_period('M'))['tono'].apply(
            lambda x: (x == 'positivo').sum() / len(x) * 100).rename(f"% Positivas {cand2}")
        pct_pos = pd.concat([p1, p2], axis=1).fillna(0)
        pct_pos.index = pct_pos.index.to_timestamp()

        fig4 = px.line(pct_pos, markers=True)
        fig4.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300)
        st.plotly_chart(fig4, use_container_width=True)

# === TAB 3: Análisis de Narrativa ===
with tab3:
    st.header("Análisis de Narrativa del Candidato")

    candidate_narr = st.selectbox("Selecciona un candidato", df['index'].unique(), key='narr')
    filtered_narr = df[df['index'] == candidate_narr]

    st.subheader("Nube de palabras (solo palabras con frecuencia mayor a 5)")
    text = " ".join(filtered_narr["articleBody_clean"].dropna())

    tokens = re.findall(r'\b\w+\b', text.lower())
    word_freq = Counter(tokens)
    filtered_freq = {word: count for word, count in word_freq.items() if count > 5}

    if filtered_freq:
        wordcloud = WordCloud(width=900, height=400, background_color='white').generate_from_frequencies(filtered_freq)
        fig5, ax = plt.subplots(figsize=(10, 3))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig5)
    else:
        st.warning("No hay palabras con frecuencia mayor a 5 para este candidato.")