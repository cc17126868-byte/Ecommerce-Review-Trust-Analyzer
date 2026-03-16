import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
from transformers import pipeline
from collections import Counter
import re

# ==============================
# Model Loading
# ==============================

@st.cache_resource
def load_models():
    """Load HuggingFace models."""

    fake_detector = pipeline(
        "text-classification",
        model="JerryJJJJJ/fake-review-detector-distilbert"
    )

    sentiment_model = pipeline(
        "sentiment-analysis",
        model="JerryJJJJJ/sentiment-amazon-roberta"
    )

    summarizer = pipeline(
        "text-generation",
        model="JerryJJJJJ/review-summarization-flan-t5"
    )

    return fake_detector, sentiment_model, summarizer


# ==============================
# Fake Review Detection
# ==============================

def detect_fake_reviews(reviews, model):

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
# Review Summary
# ==============================

def generate_summary(reviews, model):

    if len(reviews) == 0:
        return "No real reviews available."

    sample_reviews = random.sample(reviews, min(10, len(reviews)))

    combined_text = " ".join(sample_reviews[:5])

    prompt = f"""
    Instruction: Summarize the following customer reviews into one short sentence.

    Reviews:
    {combined_text}

    Summary:
    """

    result = model(
        prompt,
        max_new_tokens=50,
        do_sample=False
    )

    summary = result[0]["generated_text"]

    if "Summary:" in summary:
        summary = summary.split("Summary:")[-1]

    return summary.strip()


# ==============================
# Keyword Extraction
# ==============================

def extract_keywords(reviews):

    text = " ".join(reviews).lower()

    words = re.findall(r'\b[a-z]{4,}\b', text)

    stopwords = {
        "this","that","with","have","very","from","they","were","would",
        "there","their","about","after","before","really"
    }

    filtered = [w for w in words if w not in stopwords]

    common = Counter(filtered).most_common(10)

    return common


# ==============================
# Visualization
# ==============================

def plot_pie(labels, values, title, colors):

    fig, ax = plt.subplots()

    ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors)

    ax.set_title(title)

    return fig


# ==============================
# Dataset Processing
# ==============================

def process_dataset(df, fake_model, sentiment_model, summarizer):

    reviews = df["review_body"].dropna().tolist()

    fake_stats, real_reviews = detect_fake_reviews(reviews, fake_model)

    sentiment_stats = analyze_sentiment(real_reviews, sentiment_model)

    summary = generate_summary(real_reviews, summarizer)

    keywords = extract_keywords(real_reviews)

    return fake_stats, sentiment_stats, summary, keywords


# ==============================
# Single Review Analysis
# ==============================

def analyze_single_review(review, fake_model, sentiment_model, summarizer):

    fake_result = fake_model(review)[0]

    output = {"fake": fake_result}

    if fake_result["label"] not in ["FAKE", "LABEL_0"]:

        sentiment = sentiment_model(review)[0]

        summary = generate_summary([review], summarizer)

        output["sentiment"] = sentiment
        output["summary"] = summary

    return output


# ==============================
# Main UI
# ==============================

def main():

    st.set_page_config(
        page_title="E-Commerce Review Trust Analyzer",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 E-Commerce Review Trust & Feedback Analyzer")

    st.markdown(
    """
    Detect **fake reviews**, analyze **customer sentiment**, and generate **automatic feedback insights**.
    """
    )

    fake_model, sentiment_model, summarizer = load_models()

    tab1, tab2 = st.tabs(["Single Review", "Dataset Analysis"])

    # ==================================================
    # Single Review
    # ==================================================

    with tab1:

        st.header("Single Review Analysis")

        review = st.text_area("Enter review text")

        if st.button("Analyze Review"):

            result = analyze_single_review(
                review,
                fake_model,
                sentiment_model,
                summarizer
            )

            col1, col2 = st.columns(2)

            col1.subheader("Fake Review Detection")
            fake_label = result["fake"]["label"]
            fake_score = result["fake"]["score"]

            if fake_label in ["FAKE", "LABEL_0"]:
                status = "⚠️ Potential Fake Review"
            else:
                status = "✅ Authentic Review"

            col1.subheader("Review Authenticity")

            col1.write(status)

            col1.caption(f"Confidence: {round(fake_score*100,2)}%")

            if "sentiment" in result:

                col2.subheader("Sentiment")
                sent_label = result["sentiment"]["label"]
                sent_score = result["sentiment"]["score"]
                
                if sent_label in ["POSITIVE", "LABEL_1"]:
                    sentiment = "😊 Positive"
                else:
                    sentiment = "😞 Negative"
                col2.subheader("Customer Sentiment")
                col2.write(sentiment)
                col2.caption(f"Confidence: {round(sent_score*100,2)}%")

                st.subheader("Review Summary")
                st.write(result["summary"])

            else:

                st.warning("Review likely fake. Sentiment skipped.")


    # ==================================================
    # Dataset Analysis
    # ==================================================

    with tab2:

        st.header("Dataset Review Analysis")

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file:

            df = pd.read_csv(uploaded_file)

            with st.expander("Dataset Preview"):
                st.dataframe(df.head())

            progress = st.progress(0)

            fake_stats, sentiment_stats, summary, keywords = process_dataset(
                df,
                fake_model,
                sentiment_model,
                summarizer
            )

            progress.progress(100)

            st.subheader("Key Metrics")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Reviews", fake_stats["total"])
            col2.metric("Fake Reviews", fake_stats["fake"])
            col3.metric("Real Reviews", fake_stats["real"])

            trust_score = round((1 - fake_stats["fake"] / fake_stats["total"]) * 100, 2)

            st.metric("Store Trust Score", f"{trust_score}/100")

            if trust_score > 80:
                st.success("Low Fake Review Risk")

            elif trust_score > 60:
                st.warning("Moderate Fake Review Risk")

            else:
                st.error("High Fake Review Risk")


            st.subheader("Review Authenticity")

            fig1 = plot_pie(
                ["Fake", "Real"],
                [fake_stats["fake"], fake_stats["real"]],
                "Fake vs Real Reviews",
                ["#FF9800", "#2196F3"]
            )

            st.pyplot(fig1)


            st.subheader("Customer Sentiment")

            fig2 = plot_pie(
                ["Positive", "Negative"],
                [sentiment_stats["positive"], sentiment_stats["negative"]],
                "Sentiment Distribution",
                ["#4CAF50", "#FF5252"]
            )

            st.pyplot(fig2)


            st.subheader("Top Customer Feedback Keywords")

            kw_df = pd.DataFrame(keywords, columns=["Keyword", "Frequency"])

            st.table(kw_df)


            st.subheader("Customer Feedback Summary")

            st.write(summary)


            st.subheader("Business Insight")

            if sentiment_stats["positive"] > sentiment_stats["negative"]:

                st.info(
                    "Customers generally express positive sentiment toward the product."
                )

            else:

                st.warning(
                    "Customer sentiment appears negative. Merchants should investigate product issues."
                )


# ==============================
# Entry
# ==============================

if __name__ == "__main__":
    main()
