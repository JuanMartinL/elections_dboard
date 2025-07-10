import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('dataout/candidates_news.csv')

df['date_published'] = pd.to_datetime(df['date_published'])
df['date_published'] = df['date_published'].dt.tz_localize(None)

# Layout and Tabs
st.set_page_config(layout="wide")
st.title("Escucha Social - Análisis de Candidatos")

tab1, tab2, tab3 = st.tabs(["Análisis Candidatos", "Comparativo", "Análisis de Narrativa"])

# === TAB 1: Análisis de Candidatos ===
with tab1:
    st.header("Análisis de un Candidato")
    col1, col2 = st.columns([3, 1])

    with col1:
        candidate = st.selectbox("Selecciona un candidato", df['index'].unique())
        # Extract unique months from the dataset
        df['month_period'] = df['date_published'].dt.to_period('M')
        available_months = sorted(df['month_period'].unique())

        # Monthly slicer range
        start_month, end_month = st.select_slider(
            "Selecciona rango de meses",
            options=available_months,
            value=(available_months[0], available_months[-1]),
            format_func=lambda x: x.strftime("%b %Y")
        )


    with col2:
        # Convert Periods to timestamps
        start_date = start_month.to_timestamp()
        end_date = end_month.to_timestamp(how='end')

        filtered = df[
            (df['index'] == candidate) &
            (df['date_published'] >= start_date) &
            (df['date_published'] <= end_date)
        ]
        st.metric("Menciones Totales", len(filtered))
        st.metric("Positivas", (filtered['tono'] == 'positivo').sum())
        st.metric("Negativas", (filtered['tono'] == 'negativo').sum())

    st.subheader("Serie histórica de menciones")
    mentions_ts = (
        filtered
        .groupby(filtered['date_published'].dt.to_period('M'))
        .size()
        .reset_index(name='Menciones')
    )
    mentions_ts['date_published'] = mentions_ts['date_published'].dt.to_timestamp()
    fig1 = px.line(mentions_ts, x='date_published', y='Menciones', markers=True)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Tono de las noticias a lo largo del tiempo")
    sentiment = filtered.groupby([filtered['date_published'].dt.to_period('M'), 'tono']).size().unstack().fillna(0)
    sentiment = sentiment.div(sentiment.sum(axis=1), axis=0) * 100
    sentiment.index = sentiment.index.to_timestamp()
    fig2 = px.area(sentiment, title="Distribución de sentimientos (%)")
    st.plotly_chart(fig2, use_container_width=True)

# === TAB 2: Comparativo ===
with tab2:
    st.header("Comparativo entre Candidatos")
    col1, col2 = st.columns(2)

    with col1:
        cand1 = st.selectbox("Candidato A", df['index'].unique(), key='cand1')
    with col2:
        cand2 = st.selectbox("Candidato B", df['index'].unique(), key='cand2')

    df1 = df[df['index'] == cand1]
    df2 = df[df['index'] == cand2]

    st.subheader("Comparación de menciones mensuales")
    if cand1 == cand2:
        st.warning("Por favor selecciona dos candidatos distintos para el comparativo.")
    else:
        m1 = df1.groupby(df1['date_published'].dt.to_period('M')).size().rename(cand1)
        m2 = df2.groupby(df2['date_published'].dt.to_period('M')).size().rename(cand2)
        comparison = pd.concat([m1, m2], axis=1).fillna(0)
        comparison.index = comparison.index.to_timestamp()

        fig3 = px.line(comparison, markers=True)
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Comparación de % de noticias positivas")
        p1 = df1.groupby(df1['date_published'].dt.to_period('M'))['tono'].apply(
            lambda x: (x == 'positivo').sum() / len(x) * 100).rename(f"% Positivas {cand1}")
        p2 = df2.groupby(df2['date_published'].dt.to_period('M'))['tono'].apply(
            lambda x: (x == 'positivo').sum() / len(x) * 100).rename(f"% Positivas {cand2}")
        pct_pos = pd.concat([p1, p2], axis=1).fillna(0)
        pct_pos.index = pct_pos.index.to_timestamp()
        fig4 = px.line(pct_pos, markers=True)
        st.plotly_chart(fig4, use_container_width=True)

# === TAB 3: Análisis de Narrativa ===
with tab3:
    st.header("Análisis de Narrativa del Candidato")
    candidate_narr = st.selectbox("Selecciona un candidato para la narrativa", df['index'].unique(), key='narr')
    filtered_narr = df[df['index'] == candidate_narr]

    st.subheader("🔤 Nube de palabras (cuerpo del artículo)")
    text = " ".join(filtered_narr["articleBody_clean"].dropna())
    if text:
        wordcloud = WordCloud(width=900, height=400, background_color='white').generate(text)
        fig5, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig5)
    else:
        st.warning("No hay texto disponible para este candidato.")