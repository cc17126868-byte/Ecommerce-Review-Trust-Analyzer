import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import Counter
import re

# ==============================
# Model Loading
# ==============================

@st.cache_resource
def load_models():

    fake_detector = pipeline(
        "text-classification",
        model="JerryJJJJJ/Fake-review-detector-distilbert-V2"
    )

    sentiment_model = pipeline(
        "sentiment-analysis",
        model="JerryJJJJJ/sentiment-amazon-roberta"
    )

    model_name = "JerryJJJJJ/review-summarization-flan-t5"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarization_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return fake_detector, sentiment_model, tokenizer, summarization_model


# ==============================
# Fake Review Detection
# ==============================

def detect_fake_reviews(reviews, model):

    results = model(reviews)

    fake_count = 0
    real_reviews = []

    for review, result in zip(reviews, results):

        label = result["label"]

        if label in ["FAKE", "LABEL_1"]:
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

    if len(reviews) == 0:
        return {"positive": 0, "negative": 0}
    
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

def generate_summary(reviews, tokenizer, model):

    if len(reviews) == 0:
        return "No authentic reviews available."

    sample_reviews = random.sample(reviews, min(5, len(reviews)))

    combined_text = " ".join(sample_reviews)

    inputs = tokenizer(
        combined_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_length=40
    )

    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary


# ==============================
# Keyword Extraction
# ==============================

def extract_keywords(reviews):

    if len(reviews) == 0:
        return []
    
    text = " ".join(reviews).lower()

    words = re.findall(r'\b[a-z]{4,}\b', text)

    stopwords = {
        "this","that","with","have","very","from","they","were","would",
        "there","their","about","after","before","really","just","what",
        "when","can","will","more","than","been","had","did","does","has"
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

def process_dataset(df, fake_model, sentiment_model, tokenizer, summarization_model):
    """
    Process dataset with support for multiple review column names
    """
    # Define possible review column names
    possible_columns = [
        'review_body',
        'review_text',
        'text',
        'comment',
        'review',
        'feedback',
        'content',
        'review_content',
        'customer_review',
        'product_review',
        'review_description',
        'comment_text',
        'user_review',
        'review_comment',
        'body'
    ]
    
    # Find existing column
    review_column = None
    for col in possible_columns:
        if col in df.columns:
            review_column = col
            break
    
    # If no standard column found, try using the first text column
    if review_column is None:
        # Get all possible text columns
        text_columns = df.select_dtypes(include=['object']).columns
        
        # Exclude potential non-review columns
        exclude_patterns = ['id', 'date', 'time', 'rating', 'score', 'helpful', 'vote']
        candidate_columns = [
            col for col in text_columns 
            if not any(pattern in col.lower() for pattern in exclude_patterns)
        ]
        
        if len(candidate_columns) > 0:
            # Use the first eligible column
            review_column = candidate_columns[0]
            st.info(f"✅ Automatically detected review column: '{review_column}'")
        else:
            st.error("""
            ❌ No review column found. Please ensure your CSV file contains a text column for reviews.
            
            Supported column names include: review_body, review_text, text, comment, review, feedback, content, etc.
            
            Your CSV file contains the following columns:
            """)
            
            # Show available columns for reference
            st.write(list(df.columns))
            
            return None, None
    
    # Get review data
    reviews = df[review_column].dropna().astype(str).tolist()
    
    if len(reviews) == 0:
        st.warning("⚠️ No valid review data found.")
        return None, None
    
    # Show processing information
    st.success(f"✅ Successfully loaded {len(reviews)} reviews. Analyzing...")
    
    # Analyze fake reviews
    fake_stats, real_reviews = detect_fake_reviews(reviews, fake_model)
    
    # Analyze sentiment
    sentiment_stats = analyze_sentiment(real_reviews, sentiment_model)
      
    return fake_stats, sentiment_stats


# ==============================
# Single Review Analysis
# ==============================

def analyze_single_review(review, fake_model, sentiment_model, tokenizer, summarization_model):

    fake_result = fake_model(review)[0]

    output = {"fake": fake_result}

    if fake_result["label"] == "LABEL_1":
        pass
    else:
        sentiment = sentiment_model(review)[0]

        summary = generate_summary([review], tokenizer, summarization_model)

        output["sentiment"] = sentiment
        output["summary"] = summary

    return output


# ==============================
# Main UI
# ==============================

def main():

    st.set_page_config(
        page_title="Review Trust Analyzer",
        page_icon="🔍",
        layout="wide"
    )

    # ==================================================
    # Sidebar - Model Information
    # ==================================================
    
    with st.sidebar:
        st.header("📌 About This Tool")
        st.markdown("""
        ### Three Fine-Tuned Models
        
        | Task | Model |
        |------|-------|
        | 🔍 Fake Review Detection | DistilBERT |
        | 😊 Sentiment Analysis | RoBERTa |
        | 📝 Review Summarization | FLAN-T5 |
        
        ---
        
        ### Label Definitions
        
        | Model Output | Meaning |
        |--------------|---------|
        | **LABEL_1 / FAKE** | 🤖 **AI-Generated / Fake Review** |
        | **LABEL_0 / REAL** | 👤 **Human-Written / Authentic Review** |
        
        ---
        
        ### How to Use
        
        **Single Review:** Enter text and click analyze
        
        **Batch Analysis:** Upload CSV with review column
        
        Supported column names: `text`, `review`, `content`, `comment`, `review_body`, etc.
        
        ---
        
        ### Why It Matters
        
        AI-generated fake reviews can:
        - Mislead customers
        - Damage business reputation
        - Skew product ratings
        
        This tool helps e-commerce merchants maintain **review integrity**.
        """)
        
        st.divider()
        st.caption("🚀 Powered by DistilBERT | RoBERTa | FLAN-T5")

    # ==================================================
    # Main Title
    # ==================================================
    
    st.title("🔍 Review Trust Analyzer")
    
    st.markdown(
        """
        Detect **AI-generated fake reviews**, analyze **customer sentiment**, and generate **automatic feedback insights**.
        """
    )
    
    st.caption("⚡ Powered by fine-tuned DistilBERT, RoBERTa, and FLAN-T5 models")

    # ==================================================
    # Model Loading with Status
    # ==================================================
    
    with st.spinner("Loading AI models... This may take a moment."):
        fake_model, sentiment_model, tokenizer, summarization_model = load_models()
    
    st.success("✅ Models loaded successfully!")

    # ==================================================
    # Tabs
    # ==================================================
    
    tab1, tab2 = st.tabs(["🔍 Single Review", "📊 Batch Analysis"])

    # ==================================================
    # Tab 1: Single Review
    # ==================================================

    with tab1:

        st.header("🔍 Single Review Analysis")
        
        st.caption("Enter a customer review to check its authenticity, sentiment, and generate a summary.")
        
        review = st.text_area(
            "Review text",
            height=150,
            placeholder="Paste a customer review here... e.g., This product exceeded my expectations! The quality is outstanding and delivery was fast."
        )
        
        if st.button("🔍 Analyze Review", type="primary", use_container_width=False):
            
            if not review.strip():
                st.warning("⚠️ Please enter a review to analyze.")
            else:
                with st.spinner("Analyzing review..."):
                    result = analyze_single_review(
                        review,
                        fake_model,
                        sentiment_model,
                        tokenizer, 
                        summarization_model
                    )
                
                st.divider()
                
                # Results layout
                col1, col2 = st.columns(2)
                
                # Column 1: Authenticity Result
                with col1:
                    st.subheader("🔍 Review Authenticity")
                    fake_label = result["fake"]["label"]
                    fake_score = result["fake"]["score"]
                    
                    if fake_label in ["FAKE", "LABEL_1"]:
                        st.error("🤖 **AI-Generated / Fake Review**")
                        st.caption(f"Confidence: {round(fake_score*100,2)}%")
                        st.info("⚠️ This review was likely created by a computer or automated system.")
                    else:
                        st.success("👤 **Human-Written / Authentic Review**")
                        st.caption(f"Confidence: {round(fake_score*100,2)}%")
                        st.info("✅ This review appears to be written by a real customer.")
                
                # Column 2: Sentiment Result (only for authentic reviews)
                if "sentiment" in result:
                    with col2:
                        st.subheader("😊 Customer Sentiment")
                        sent_label = result["sentiment"]["label"]
                        sent_score = result["sentiment"]["score"]
                        
                        if sent_label in ["POSITIVE", "LABEL_1"]:
                            st.success("😊 **Positive**")
                        else:
                            st.error("😞 **Negative**")
                        st.caption(f"Confidence: {round(sent_score*100,2)}%")
                
                # Summary (only for authentic reviews)
                if "sentiment" in result:
                    st.subheader("📝 Review Summary")
                    st.info(result["summary"])
                else:
                    st.warning("ℹ️ Sentiment analysis and summarization skipped for AI-generated reviews.")
                
                st.divider()

    # ==================================================
    # Tab 2: Dataset Analysis
    # ==================================================

    with tab2:

        st.header("📊 Batch Review Analysis")
        
        st.caption("""
        Upload a CSV file containing customer reviews. The tool will analyze all reviews and provide:
        - AI-generated review detection
        - Sentiment analysis (for authentic reviews only)
        - Business insights and recommendations
        """)
        
        st.info("💡 **Supported review column names:** `text`, `review`, `content`, `comment`, `review_body`, `feedback`, etc.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_file:
            
            df = pd.read_csv(uploaded_file)
            
            with st.expander("📋 Dataset Preview", expanded=True):
                st.dataframe(df.head())
                st.caption(f"📊 Total rows: {len(df)} | Columns: {list(df.columns)}")
            
            if st.button("🚀 Start Analysis", type="primary", use_container_width=False):
                
                # Progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📂 Loading reviews...")
                progress_bar.progress(20)
                
                # Process dataset
                result = process_dataset(
                    df,
                    fake_model,
                    sentiment_model,
                    tokenizer,
                    summarization_model
                )
                
                if result[0] is None:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ Analysis failed. Please check your CSV format.")
                else:
                    fake_stats, sentiment_stats = result
                    
                    status_text.text("📊 Generating insights...")
                    progress_bar.progress(60)
                    
                    # ===== Results Display =====
                    
                    st.divider()
                    st.subheader("📈 Analysis Results")
                    
                    # Key Metrics Row
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Reviews", fake_stats["total"])
                    with col2:
                        st.metric("🤖 AI-Generated", fake_stats["fake"], delta=f"-{fake_stats['fake_percentage']}%")
                    with col3:
                        st.metric("👤 Human-Written", fake_stats["real"])
                    with col4:
                        trust_score = round((1 - fake_stats["fake"] / fake_stats["total"]) * 100, 2)
                        st.metric("Trust Score", f"{trust_score}%")
                    
                    # Trust Score Feedback
                    if trust_score > 80:
                        st.success(f"✅ **Trust Score: {trust_score}/100** - Low risk of fake reviews")
                    elif trust_score > 60:
                        st.warning(f"⚠️ **Trust Score: {trust_score}/100** - Moderate risk, monitor patterns")
                    else:
                        st.error(f"❌ **Trust Score: {trust_score}/100** - High risk, investigate immediately")
                    
                    status_text.text("📈 Creating visualizations...")
                    progress_bar.progress(80)
                    
                    # Charts Row
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Review Authenticity")
                        fig1 = plot_pie(
                            ["🤖 AI-Generated", "👤 Human-Written"],
                            [fake_stats["fake"], fake_stats["real"]],
                            "AI-Generated vs Human-Written Reviews",
                            ["#FF9800", "#2196F3"]
                        )
                        st.pyplot(fig1)
                    
                    with col2:
                        st.subheader("Customer Sentiment")
                        fig2 = plot_pie(
                            ["😊 Positive", "😞 Negative"],
                            [sentiment_stats["positive"], sentiment_stats["negative"]],
                            "Sentiment Distribution (Authentic Reviews Only)",
                            ["#4CAF50", "#FF5252"]
                        )
                        st.pyplot(fig2)
                    
                    # Business Insights
                    st.subheader("💡 Business Insights")
                    
                    # AI-Generated Review Risk Analysis
                    if fake_stats["fake_percentage"] > 30:
                        st.error(f"""
                        **⚠️ HIGH AI-GENERATED REVIEW RISK**  
                        {fake_stats['fake_percentage']}% of reviews appear to be computer-generated.
                        
                        **Recommended Actions:**
                        - Review your review moderation process
                        - Consider implementing CAPTCHA for review submission
                        - Investigate suspicious review patterns (same IP, repetitive content)
                        - Flag and remove confirmed fake reviews
                        """)
                    elif fake_stats["fake_percentage"] > 10:
                        st.warning(f"""
                        **⚠️ MODERATE AI-GENERATED REVIEW RISK**  
                        {fake_stats['fake_percentage']}% of reviews appear to be computer-generated.
                        
                        **Recommended Actions:**
                        - Monitor review patterns for anomalies
                        - Strengthen review verification process
                        - Consider adding review confirmation emails
                        """)
                    else:
                        st.success(f"""
                        **✅ LOW AI-GENERATED REVIEW RISK**  
                        Only {fake_stats['fake_percentage']}% of reviews are flagged as computer-generated.
                        
                        Your review system appears healthy with mostly authentic customer feedback.
                        """)
                    
                    # Sentiment Analysis Insight
                    if sentiment_stats["positive"] > sentiment_stats["negative"]:
                        st.info(f"""
                        **📈 POSITIVE CUSTOMER SENTIMENT**  
                        {sentiment_stats['positive']} out of {fake_stats['real']} authentic reviews are positive.
                        
                        Customers generally express positive sentiment. This is a good sign for product quality and customer satisfaction.
                        """)
                    else:
                        st.warning(f"""
                        **📉 NEGATIVE CUSTOMER SENTIMENT**  
                        {sentiment_stats['negative']} out of {fake_stats['real']} authentic reviews are negative.
                        
                        Customer sentiment appears negative. Merchants should investigate product issues and address common complaints.
                        """)
                    
                    
                    # Complete
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    st.success("✅ Analysis completed successfully!")


# ==============================
# Entry
# ==============================

if __name__ == "__main__":
    main()
