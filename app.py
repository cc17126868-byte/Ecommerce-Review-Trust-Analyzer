import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
from transformers import pipeline

# ==============================
# Model Loading
# ==============================

@st.cache_resource
def load_models():
    """Load all HuggingFace models."""

    fake_detector = pipeline(
        "text-classification",
        model="JerryJJJJJ/fake-review-detector-distilbert"
    )

    sentiment_model = pipeline(
        "sentiment-analysis",
        model="JerryJJJJJ/sentiment-amazon-roberta"
    )

    summarizer = pipeline(
        "summarization",
        model="JerryJJJJJ/review-summarization-flan-t5"
    )

    return fake_detector, sentiment_model, summarizer


# ==============================
# Fake Review Detection
# ==============================

def detect_fake_reviews(reviews, model):
    """Detect fake reviews and return statistics."""

    results = model(reviews)

    fake_count = 0
    real_reviews = []

    for review, result in zip(reviews, results):

        label = result["label"]

        if label in ["FAKE", "LABEL_0"]:
            fake_count += 1
        else:
            real_reviews.append(review)

    total = len(reviews)

    stats = {
        "total": total,
        "fake": fake_count,
        "real": len(real_reviews),
        "fake_percentage": round(fake_count / total * 100, 2)
    }

    return stats, real_reviews


# ==============================
# Sentiment Analysis
# ==============================

def analyze_sentiment(reviews, model):
    """Analyze sentiment of real reviews."""

    results = model(reviews)

    positive = 0
    negative = 0

    for r in results:

        label = r["label"]

        if label in ["POSITIVE", "LABEL_1"]:
            positive += 1
        else:
            negative += 1

    stats = {
        "positive": positive,
        "negative": negative
    }

    return stats


# ==============================
# Review Summarization
# ==============================

def generate_summary(reviews, model):
    """Generate summary from sample reviews."""

    if len(reviews) == 0:
        return "No real reviews available for summarization."

    sample_reviews = random.sample(reviews, min(20, len(reviews)))

    combined_text = " ".join(sample_reviews)

    result = model(
        combined_text,
        max_length=80,
        min_length=20,
        do_sample=False
    )

    return result[0]["generated_text"]


# ==============================
# Visualization
# ==============================

def plot_pie_chart(labels, values, title):
    """Create pie chart."""

    fig, ax = plt.subplots()

    ax.pie(values, labels=labels, autopct="%1.1f%%")

    ax.set_title(title)

    return fig


# ==============================
# Single Review Analysis
# ==============================

def analyze_single_review(review, fake_model, sentiment_model, summarizer):

    fake_result = fake_model(review)[0]

    output = {
        "fake": fake_result
    }

    if fake_result["label"] not in ["FAKE", "LABEL_0"]:

        sentiment = sentiment_model(review)[0]

        summary = summarizer(
            "summarize: " + review,
            max_length=40,
            min_length=5,
            do_sample=False
        )[0]["generated_text"]

        output["sentiment"] = sentiment
        output["summary"] = summary

    return output


# ==============================
# Dataset Processing
# ==============================

def process_dataset(df, fake_model, sentiment_model, summarizer):

    reviews = df["review_body"].dropna().tolist()

    fake_stats, real_reviews = detect_fake_reviews(reviews, fake_model)

    sentiment_stats = analyze_sentiment(real_reviews, sentiment_model)

    summary = generate_summary(real_reviews, summarizer)

    return fake_stats, sentiment_stats, summary


# ==============================
# UI
# ==============================

def main():

    st.set_page_config(
        page_title="E-Commerce Review Trust Analyzer",
        layout="wide"
    )

    st.title("E-Commerce Review Trust & Feedback Analyzer")

    st.write(
        "This application detects fake reviews, analyzes customer sentiment, "
        "and summarizes genuine feedback to help merchants understand product reputation."
    )

    fake_model, sentiment_model, summarizer = load_models()

    tab1, tab2 = st.tabs(["Single Review Analysis", "Dataset Analysis"])

    # ===================================
    # Single Review
    # ===================================

    with tab1:

        st.header("Single Review Analysis")

        review = st.text_area("Enter a review:")

        if st.button("Analyze Review"):

            if review.strip() == "":
                st.warning("Please enter a review.")

            else:

                result = analyze_single_review(
                    review,
                    fake_model,
                    sentiment_model,
                    summarizer
                )

                st.subheader("Fake Review Detection")

                st.write(result["fake"])

                if "sentiment" in result:

                    st.subheader("Sentiment")

                    st.write(result["sentiment"])

                    st.subheader("Review Summary")

                    st.write(result["summary"])

                else:

                    st.warning("Review likely fake. Sentiment skipped.")

    # ===================================
    # Dataset Analysis
    # ===================================

    with tab2:

        st.header("Dataset Analysis")

        st.write(
            "Upload a CSV file containing a column named **review_body**."
        )

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file:

            df = pd.read_csv(uploaded_file)

            if "review_body" not in df.columns:

                st.error("CSV must contain column: review_body")

            else:

                fake_stats, sentiment_stats, summary = process_dataset(
                    df,
                    fake_model,
                    sentiment_model,
                    summarizer
                )

                st.subheader("Fake Review Statistics")

                st.write(fake_stats)

                fig1 = plot_pie_chart(
                    ["Fake", "Real"],
                    [fake_stats["fake"], fake_stats["real"]],
                    "Fake vs Real Reviews"
                )

                st.pyplot(fig1)

                st.subheader("Sentiment Distribution")

                fig2 = plot_pie_chart(
                    ["Positive", "Negative"],
                    [
                        sentiment_stats["positive"],
                        sentiment_stats["negative"]
                    ],
                    "Sentiment Distribution"
                )

                st.pyplot(fig2)

                st.subheader("Customer Feedback Summary")

                st.write(summary)


# ==============================
# Entry Point
# ==============================

if __name__ == "__main__":
    main()
