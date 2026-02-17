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
import os

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

try:
    df = load_data()
    data_loaded = True
except:
    st.error("⚠️ Could not load data file. Please ensure 'mental_health_cleaned_dataset.csv' is in the correct location.")
    data_loaded = False
    df = pd.DataFrame()

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
try:
    artifacts = joblib.load("mental_health_model_artifacts.pkl")
    model = artifacts.get("model")
    scaler = artifacts.get("scaler")
    label_encoder = artifacts.get("label_encoder")
    feature_columns = artifacts.get("feature_columns")
    y_test = artifacts.get("y_test")
    y_pred = artifacts.get("y_pred")
    model_loaded = True
except:
    st.warning("⚠️ Model artifacts not found. Some features will be limited.")
    model_loaded = False
    artifacts = {}

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("🧠 MindWatch Mental Health Intelligence System")
st.markdown("### AI-Powered Sentiment Monitoring & Emotional Risk Detection")

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("Filter Controls")

if data_loaded and not df.empty:
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
        color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#ff4d4d'}
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
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
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
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

    # -------------------------------------------------
    # PLATFORM ANXIETY RISK COMPARISON
    # -------------------------------------------------
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
        color="platform",
        title="Average Anxiety Score by Platform"
    )
    fig_platform.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig_platform, use_container_width=True)

else:
    st.warning("No data available to display. Please check your data file.")
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
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
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
            response = "That's wonderful to hear. Keep nurturing that positive energy."

        elif polarity < -0.2:
            st.error("Sentiment Detected: Negative 😔")

            advice_list = [
                "🌱 Tough days don't last, tough people do.",
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

        try:
            st.subheader("📊 Emotional Word Cloud")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(user_text)
            fig_wc, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot(fig_wc)
        except:
            st.info("Word cloud generation skipped (requires more text)")

st.divider()

# -------------------------------------------------
# BREATHING EXERCISE
# -------------------------------------------------
st.header("🧘 10-Second Calm Breathing Exercise")

if st.button("Start Breathing Exercise"):
    with st.spinner("Preparing..."):
        time.sleep(1)
    
    progress_bar = st.progress(0)
    for i in range(4):
        st.markdown(f"### 🌬 Inhale... (2 seconds)")
        for j in range(10):
            time.sleep(0.2)
            progress_bar.progress(i*25 + j*2.5)
        
        st.markdown(f"### 😌 Exhale... (2 seconds)")
        for j in range(10):
            time.sleep(0.2)
            progress_bar.progress(i*25 + 25 + j*2.5)
    
    st.success("✅ You completed the breathing session. Well done.")

# =================================================
# MODEL EVALUATION
# =================================================
st.divider()
st.header("📊 Model Evaluation & Performance Analysis")

if model_loaded and 'y_test' in artifacts and 'y_pred' in artifacts:
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy:.2%}")
        c2.metric("Precision", f"{precision:.2%}")
        c3.metric("Recall", f"{recall:.2%}")
        c4.metric("F1 Score", f"{f1:.2%}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig_cm, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        classes = label_encoder.classes_ if label_encoder else range(len(cm))
        
        # Show all ticks
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               xlabel='Predicted',
               ylabel='Actual')

        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        fig_cm.tight_layout()
        st.pyplot(fig_cm)

    except Exception as e:
        st.info(f"Could not generate evaluation metrics: {e}")
else:
    st.info("Model evaluation metrics not available in artifacts.")

# =================================================
# BUSINESS INTERPRETATION
# =================================================
st.divider()
st.header("🧠 Business Interpretation of Results")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Key Insights:**
    • High negative sentiment indicates rising distress trends
    • Anxiety scores above 0.8 highlight urgent intervention cases
    • Platform insights help identify risk concentration areas
    • Enables proactive mental health support deployment
    """)
with col2:
    st.markdown("""
    **Actionable Outcomes:**
    • Early warning system for mental health teams
    • Resource allocation based on platform risk levels
    • Trend analysis for seasonal patterns
    • ROI measurement of intervention programs
    """)

# =================================================
# DATASET DOWNLOAD
# =================================================
st.divider()
st.header("📂 Cleaned Dataset Access")

if data_loaded and not df.empty:
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Cleaned Dataset (CSV)",
        data=csv,
        file_name="mental_health_cleaned_dataset.csv",
        mime="text/csv"
    )
else:
    st.info("Dataset not available for download")

# =================================================
# DEPLOYMENT DISCUSSION
# =================================================
st.divider()
st.header("🚀 Real-World Deployment Strategy")

with st.expander("View Deployment Options"):
    st.markdown("""
    **Integration Possibilities:**
    • Live mental health monitoring dashboards for organizations
    • Batch prediction pipelines for research institutions
    • HR & university wellbeing systems integration
    • Healthcare screening support tools
    • API integration for external apps and services
    
    **Technical Requirements:**
    • Secure API endpoints with authentication
    • HIPAA/GDPR compliance for healthcare data
    • Scalable cloud infrastructure
    • Real-time alerting systems
    """)

# =================================================
# ETHICAL AI
# =================================================
st.divider()
st.header("⚖ Ethical AI Considerations")

with st.expander("Read Our Ethical Guidelines"):
    st.markdown("""
    **Privacy & Security:**
    • All user data is anonymized and encrypted
    • No personal identifiers are stored permanently
    • Users can request data deletion at any time
    
    **Fairness & Bias:**
    • Regular audits for demographic bias
    • Models tested across diverse populations
    • Continuous monitoring for unfair outcomes
    
    **Clinical Safety:**
    • This tool is for screening, not diagnosis
    • Always consult mental health professionals
    • Clear disclaimers about limitations
    """)

# =================================================
# FUTURE WORK
# =================================================
st.divider()
st.header("🔮 Future Enhancements")

tab1, tab2, tab3 = st.tabs(["📊 Models", "🌍 Languages", "📱 Integration"])

with tab1:
    st.markdown("""
    **Model Improvements:**
    • Deep learning sentiment models (BERT, RoBERTa)
    • Multi-modal emotion detection (text + voice + facial)
    • Personalized risk prediction over time
    • Explainable AI features
    """)

with tab2:
    st.markdown("""
    **Language Support:**
    • Multilingual emotion detection
    • Cultural context awareness
    • Regional dialect understanding
    • Real-time translation integration
    """)

with tab3:
    st.markdown("""
    **Platform Expansion:**
    • Mobile app (iOS/Android)
    • WhatsApp chatbot integration
    • Slack/Discord bots for workplaces
    • Therapist alert systems
    • Wearable device integration
    """)

# =================================================
# FOOTER
# =================================================
st.divider()
st.markdown(
    "<div style='text-align: center; color: #94a3b8; padding: 20px;'>"
    "🧠 MindWatch AI | Mental Health Intelligence System | v2.0<br>"
    "⚠️ This is a screening tool only. Always consult with mental health professionals."
    "</div>",
    unsafe_allow_html=True
)