import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
import string

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------------------------
# LOAD CSS
# --------------------------------------------------

with open("styles.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

simple_rnn_model = load_model("simple_rnn_model.h5")
lstm_model = load_model("lstm_model.h5")
gru_model = load_model("gru_model.h5")

# --------------------------------------------------
# LOAD TOKENIZER
# --------------------------------------------------

with open("tokenizer (1).pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 250

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    return text


def prepare_input(review):

    review = preprocess_text(review)

    sequence = tokenizer.texts_to_sequences([review])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    return padded


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_review(model, review):

    processed = prepare_input(review)

    prediction = model.predict(
        processed,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive 😊"
        if prediction >= 0.5
        else "Negative 😞"
    )

    confidence = (
        prediction * 100
        if prediction >= 0.5
        else (1 - prediction) * 100
    )

    positive_prob = prediction * 100
    negative_prob = (1 - prediction) * 100

    return (
        sentiment,
        confidence,
        positive_prob,
        negative_prob
    )


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="main-title">
    🎬 Movie Review Sentiment Analysis System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
    Deep Learning Based Sentiment Classification
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

selected_model = st.radio(
    "Select Model",
    ["SimpleRNN", "LSTM", "GRU"],
    horizontal=True
)

# --------------------------------------------------
# REVIEW INPUT
# --------------------------------------------------

review_text = st.text_area(
    "Enter your movie review here...",
    height=200
)

# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------

if st.button("Analyze Review"):

    if review_text.strip() == "":

        st.warning(
            "Please enter a movie review."
        )

    else:

        if selected_model == "SimpleRNN":
            model = simple_rnn_model

        elif selected_model == "LSTM":
            model = lstm_model

        else:
            model = gru_model

        (
            sentiment,
            confidence,
            pos_prob,
            neg_prob
        ) = predict_review(
            model,
            review_text
        )

        # --------------------------------------
        # RESULT
        # --------------------------------------

        st.success(
            f"🎯 Sentiment: {sentiment}"
        )

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

        # --------------------------------------
        # PROBABILITY METRICS
        # --------------------------------------

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Positive Probability",
                f"{pos_prob:.2f}%"
            )

        with col2:
            st.metric(
                "Negative Probability",
                f"{neg_prob:.2f}%"
            )

        # --------------------------------------
        # PROGRESS BARS
        # --------------------------------------

        st.subheader(
            "Probability Distribution"
        )

        st.write(
            f"Positive : {pos_prob:.2f}%"
        )

        st.progress(
            int(pos_prob)
        )

        st.write(
            f"Negative : {neg_prob:.2f}%"
        )

        st.progress(
            int(neg_prob)
        )

        # --------------------------------------
        # CHART
        # --------------------------------------

        st.subheader(
            "Confidence Chart"
        )

        chart_df = pd.DataFrame({
            "Class": [
                "Positive",
                "Negative"
            ],
            "Probability": [
                pos_prob,
                neg_prob
            ]
        })

        fig, ax = plt.subplots(
            figsize=(6,4)
        )

        ax.bar(
            chart_df["Class"],
            chart_df["Probability"]
        )

        ax.set_ylabel(
            "Probability (%)"
        )

        ax.set_title(
            "Sentiment Probability"
        )

        st.pyplot(fig)

        st.markdown("---")

        # --------------------------------------
        # COMPARE ALL MODELS
        # --------------------------------------

        st.subheader(
            "⚡ Compare Predictions From All Models"
        )

        results = []

        for model_name, model in [

            ("SimpleRNN", simple_rnn_model),

            ("LSTM", lstm_model),

            ("GRU", gru_model)

        ]:

            (
                sentiment,
                confidence,
                pos_prob,
                neg_prob
            ) = predict_review(
                model,
                review_text
            )

            results.append([
                model_name,
                sentiment,
                round(confidence, 2)
            ])

        comparison_df = pd.DataFrame(
            results,
            columns=[
                "Model",
                "Predicted Sentiment",
                "Confidence (%)"
            ]
        )

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
    Movie Review Sentiment Analysis using
    SimpleRNN, LSTM and GRU
    </div>
    """,
    unsafe_allow_html=True
)