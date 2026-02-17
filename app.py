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
.stApp {background-color: #0f172a; color: #ffffff;}
.main {background-color: #0f172a;}
section[data-testid="stSidebar"] {background-color: #1e293b;}
h1, h2, h3, h4, h5, h6, p, span, div, label {color: #ffffff !important;}
[data-testid="stMetricValue"] {color: #ffffff !important;}
[data-testid="stMetricLabel"] {color: #94a3b8 !important;}
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
model = artifacts.get("model")
scaler = artifacts.get("scaler")
label_encoder = artifacts.get("label_encoder")
feature_columns = artifacts.get("feature_columns")

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
    title='Sentiment Distribution'
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
# EMOTIONAL ANALYZER
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

        st.plotly_chart(sentiment_gauge(polarity), use_container_width=True)

        risk_score = abs(polarity)
        st.metric("Emotional Risk Score", f"{risk_score:.2f}")

        if polarity > 0.2:
            st.success("Sentiment Detected: Positive 😊")
            response = "That’s wonderful to hear. Keep nurturing that positive energy."

        elif polarity < -0.2:
            st.error("Sentiment Detected: Negative 😔")

            advice_list = [
                "🌱 Tough days don’t last, tough people do.",
                "💙 It's okay to not be okay.",
                "🌤 Small progress is still progress.",
                "🫂 You are not alone.",
                "🌟 Every storm runs out of rain."
            ]

            response = random.choice(advice_list)

            if polarity < -0.5:
                st.error("⚠️ High emotional distress detected. Consider speaking to a professional.")

        else:
            st.warning("Sentiment Detected: Neutral 😐")
            response = "Thank you for sharing. Remember to check in with yourself regularly."

        with st.chat_message("user"):
            st.write(user_text)

        with st.chat_message("assistant"):
            st.write(response)

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

# =================================================
# MODEL EVALUATION
# =================================================
st.divider()
st.header("📊 Model Evaluation & Performance Analysis")

try:
    y_test = artifacts["y_test"]
    y_pred = artifacts["y_pred"]

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy:.2f}")
    c2.metric("Precision", f"{precision:.2f}")
    c3.metric("Recall", f"{recall:.2f}")
    c4.metric("F1 Score", f"{f1:.2f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig_cm, ax = plt.subplots()
    ax.imshow(cm)

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig_cm)

except:
    st.info("Evaluation metrics not found in artifacts.")

# =================================================
# BUSINESS INTERPRETATION
# =================================================
st.divider()
st.header("🧠 Business Interpretation of Results")

st.markdown("""
• High negative sentiment indicates rising distress trends.

• Anxiety scores above 0.8 highlight urgent intervention cases.

• Platform insights help identify risk concentration areas.

• Enables proactive mental health support deployment.
""")

# =================================================
# DATASET DOWNLOAD
# =================================================
st.divider()
st.header("📂 Cleaned Dataset Access")

st.download_button(
    "Download Cleaned Dataset",
    df.to_csv(index=False),
    "mental_health_cleaned_dataset.csv",
    "text/csv"
)

# =================================================
# ADDITIONAL INSIGHTS
# =================================================
st.subheader("📱 Platform Anxiety Risk Comparison")

platform_risk = (
    df.groupby("platform")["anxiety_score"]
    .mean()
    .reset_index()
)

fig_platform = px.bar(
    platform_risk,
    x="platform",
    y="anxiety_score",
    color="platform"
)

st.plotly_chart(fig_platform, use_container_width=True)

# =================================================
# DEPLOYMENT DISCUSSION
# =================================================
st.divider()
st.header("🚀 Real-World Deployment Strategy")

st.markdown("""
• Live mental health monitoring dashboards
• Batch prediction pipelines
• HR & university wellbeing systems
• Healthcare screening support
• API integration for external apps
""")

# =================================================
# ETHICAL AI
# =================================================
st.divider()
st.header("⚖ Ethical AI Considerations")

st.markdown("""
• Protect user privacy
• Ensure data anonymity
• Avoid clinical misdiagnosis
• Monitor bias & fairness
""")

# =================================================
# FUTURE WORK
# =================================================
st.divider()
st.header("🔮 Future Enhancements")

st.markdown("""
• Deep learning sentiment models
• Multilingual emotion detection
• Voice emotion recognition
• Mobile app integration
• Therapist alert systems
""")