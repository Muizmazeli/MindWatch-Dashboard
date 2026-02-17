import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random
import time

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="MindWatch AI Dashboard", layout="wide")

# -------------------------------------------------
# CUSTOM CSS (DARK THEME)
# -------------------------------------------------
st.markdown("""
<style>

/* Main App Background */
.stApp {
    background-color: #0f172a;
    color: #ffffff;
}

/* Main Content Area */
.main {
    background-color: #0f172a;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1e293b;
}

/* Text Colors */
h1, h2, h3, h4, h5, h6, p, span, div, label {
    color: #ffffff !important;
}

/* Metric text fix */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("mental_health_cleaned_dataset.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
artifacts = joblib.load("mental_health_model_artifacts.pkl")
model = artifacts["model"]
scaler = artifacts["scaler"]
label_encoder = artifacts["label_encoder"]
feature_columns = artifacts["feature_columns"]

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("🧠 MindWatch Mental Health Intelligence System")
st.markdown("### AI-Powered Sentiment Monitoring & Emotional Risk Detection")

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("Filter Controls")

platform = st.sidebar.multiselect(
    "Platform",
    df['platform'].unique(),
    default=df['platform'].unique()
)

date_range = st.sidebar.date_input(
    "Date Range",
    [df['date'].min(), df['date'].max()]
)

filtered = df[
    (df['platform'].isin(platform)) &
    (df['date'] >= pd.to_datetime(date_range[0])) &
    (df['date'] <= pd.to_datetime(date_range[1]))
].copy()

# -------------------------------------------------
# KPI SECTION
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Posts", len(filtered))
col2.metric("Negative %",
            f"{(filtered['sentiment']=='Negative').mean()*100:.1f}%")
col3.metric("Avg Anxiety Score",
            f"{filtered['anxiety_score'].mean():.2f}")

st.divider()

# -------------------------------------------------
# SENTIMENT DISTRIBUTION
# -------------------------------------------------
sentiment_counts = (
    filtered['sentiment']
    .value_counts()
    .reset_index(name='count')
    .rename(columns={'index': 'sentiment'})
)

fig = px.bar(
    sentiment_counts,
    x='sentiment',
    y='count',
    color='sentiment',
    title='Sentiment Distribution',
    labels={'sentiment': 'Sentiment', 'count': 'Count'}
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# MONTHLY TREND
# -------------------------------------------------
filtered['year_month'] = filtered['date'].dt.to_period("M").astype(str)

time_series = (
    filtered.groupby('year_month')
    .size()
    .reset_index(name='count')
)

fig2 = px.line(
    time_series,
    x='year_month',
    y='count',
    markers=True,
    title="Monthly Post Volume"
)

st.plotly_chart(fig2, use_container_width=True)

st.divider()

# -------------------------------------------------
# HIGH RISK POSTS
# -------------------------------------------------
st.subheader("🚨 High Risk Posts (Anxiety > 0.8)")

high_risk = filtered[filtered['anxiety_score'] > 0.8][
    ['username','platform','anxiety_score','sentiment']
]

st.dataframe(high_risk, use_container_width=True)

st.divider()

# =================================================
# ADVANCED EMOTIONAL ANALYZER SECTION
# =================================================
st.header("💬 Real-Time Emotional Intelligence Analyzer")

user_text = st.text_area(
    "Describe how you're feeling today:",
    placeholder="Example: I feel exhausted and overwhelmed with everything lately."
)

def sentiment_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Emotional Polarity Score"},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.3], 'color': "#ff4d4d"},
                {'range': [-0.3, 0.3], 'color': "#ffc107"},
                {'range': [0.3, 1], 'color': "#28a745"}
            ]
        }
    ))
    return fig

if st.button("Analyze Emotion"):

    if user_text.strip() == "":
        st.warning("Please enter a sentence first.")
    else:
        blob = TextBlob(user_text)
        polarity = blob.sentiment.polarity

        # Display Gauge
        st.plotly_chart(sentiment_gauge(polarity), use_container_width=True)

        # Risk Score Calculation
        risk_score = abs(polarity)
        st.metric("Emotional Risk Score", f"{risk_score:.2f}")

        # Sentiment Classification
        if polarity > 0.2:
            sentiment_label = "Positive 😊"
            st.success(f"Sentiment Detected: {sentiment_label}")
            response = "That’s wonderful to hear. Keep nurturing that positive energy."

        elif polarity < -0.2:
            sentiment_label = "Negative 😔"
            st.error(f"Sentiment Detected: {sentiment_label}")

            advice_list = [
                "🌱 Tough days don’t last, tough people do.",
                "💙 It's okay to not be okay.",
                "🌤 Small progress is still progress.",
                "🫂 You are not alone. Consider reaching out to someone you trust.",
                "🌟 Every storm runs out of rain."
            ]

            response = random.choice(advice_list)

            if polarity < -0.5:
                st.error("⚠️ High emotional distress detected. Consider speaking to a professional.")

        else:
            sentiment_label = "Neutral 😐"
            st.warning(f"Sentiment Detected: {sentiment_label}")
            response = "Thank you for sharing. Remember to check in with yourself regularly."

        # Chat Style Response
        with st.chat_message("user"):
            st.write(user_text)

        with st.chat_message("assistant"):
            st.write(response)

        # Word Cloud
        st.subheader("📊 Emotional Word Cloud")
        wordcloud = WordCloud(background_color='white').generate(user_text)
        fig_wc, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)

st.divider()

# -------------------------------------------------
# BREATHING EXERCISE
# -------------------------------------------------
st.header("🧘 10-Second Calm Breathing Exercise")

if st.button("Start Breathing Exercise"):
    for i in range(3):
        st.markdown("### 🌬 Inhale...")
        time.sleep(2)
        st.markdown("### 😌 Exhale...")
        time.sleep(2)

st.success("You completed the breathing session. Well done.")
