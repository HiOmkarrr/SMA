# ZUDIO SOCIAL MEDIA ANALYTICS PROJECT - COMPLETE IMPLEMENTATION

# ==============================================================================
# SETUP: INSTALLING NECESSARY LIBRARIES
# ==============================================================================
# Before running, ensure you have these libraries installed.
# You can install them via pip:
# pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn plotly
# pip install vaderSentiment langdetect textblob streamlit emoji

# pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn plotly vaderSentiment langdetect textblob streamlit emoji

# Also, download necessary NLTK data by running this in a Python console:
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('punkt_tab')

# ==============================================================================
# IMPORTING LIBRARIES
# ==============================================================================
import pandas as pd
import numpy as np
import re
import string
from collections import Counter

# Data Cleaning & NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langdetect import detect, LangDetectException
import emoji

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

# Sentiment and Topic Modeling
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully.")


# ==============================================================================
# LAB 2 & 3: DATA COLLECTION, CLEANING, AND STORAGE
# ==============================================================================
# In this section, we load the data, perform rigorous cleaning and preprocessing,
# and prepare it for analysis.

def load_and_clean_data(filepath):
    """
    Loads, cleans, and preprocesses the social media data from a CSV file.
    """
    # --- LOAD DATA (LAB 2) ---
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return pd.DataFrame()

    # --- DATA CLEANING (LAB 3) ---
    print("\nStarting Data Cleaning and Preprocessing...")

    # 1. Handle Missing/Invalid Data
    df['content'].replace('[No text content]', pd.NA, inplace=True)
    df.dropna(subset=['title'], inplace=True) # Titles are crucial, drop if missing

    # 2. Feature Engineering: Combine text fields and convert timestamp
    df['full_text'] = df['title'].fillna('') + '. ' + df['content'].fillna('')
    df['created_datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    df['engagement'] = df['score'].fillna(0) + df['num_comments'].fillna(0)

    # 3. Language Detection and Filtering (Handles multi-lingual data)
    def detect_language(text):
        try:
            return detect(text)
        except LangDetectException:
            return 'unknown'

    df['language'] = df['full_text'].apply(detect_language)
    # For this analysis, we will focus on English content for consistency
    df = df[df['language'] == 'en'].copy()
    print(f"Filtered for English posts. Remaining posts: {len(df)}")

    # 4. Advanced Text Preprocessing
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user @ mentions and # hashtags
        text = re.sub(r'\@\w+|\#', '', text)
        # Demojize: convert emojis to text representation (e.g., :red_heart:)
        text = emoji.demojize(text)
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Lemmatize and remove stopwords
        cleaned_tokens = [
            lemmatizer.lemmatize(word) for word in tokens
            if word not in stop_words and len(word) > 2
        ]
        return " ".join(cleaned_tokens)

    df['cleaned_text'] = df['full_text'].apply(preprocess_text)

    # 5. Store Cleaned Data
    df.to_csv('zudio_cleaned_data.csv', index=False)
    print("Data cleaning complete. Cleaned data saved to 'zudio_cleaned_data.csv'")
    
    return df

# --- Execute Data Loading and Cleaning ---
# Make sure the CSV file is in the same directory
file_path = './Zudio_228 - zudio_posts(1).csv'
cleaned_df = load_and_clean_data(file_path)

if not cleaned_df.empty:
    print("\nSample of cleaned data:")
    print(cleaned_df[['full_text', 'cleaned_text', 'engagement', 'language']].head())


# ==============================================================================
# LAB 4: EXPLORATORY DATA ANALYSIS (EDA) AND VISUALIZATION
# ==============================================================================
# Now we explore the cleaned data to find initial patterns and insights.

def perform_eda(df):
    """
    Performs and visualizes exploratory data analysis on the dataframe.
    """
    if df.empty:
        print("DataFrame is empty. Skipping EDA.")
        return

    print("\n--- Starting Exploratory Data Analysis (EDA) ---")

    # 1. Time Series Analysis: Posts over time using Plotly
    # st.header("Discussion Volume Over Time")
    posts_over_time = df.set_index('created_datetime').resample('ME')['id'].count()
    fig_time = px.line(posts_over_time,
                       x=posts_over_time.index,
                       y='id',
                       labels={'id': 'Number of Posts', 'created_datetime': 'Month'},
                       title='Monthly Volume of Zudio-Related Posts')
    fig_time.show()

    # 2. N-gram Analysis (Bigrams)
    all_words = ' '.join(df['cleaned_text']).split()
    bigrams = (pd.Series(nltk.ngrams(all_words, 2)).value_counts())[:15]
    bigrams_df = pd.DataFrame(bigrams).reset_index().rename(columns={'index': 'bigram', 0: 'count'})
    bigrams_df['bigram'] = bigrams_df['bigram'].apply(lambda x: ' '.join(x))

    fig_bigrams = px.bar(bigrams_df,
                         x='count',
                         y='bigram',
                         orientation='h',
                         title='Top 15 Most Common Two-Word Phrases (Bigrams)')
    fig_bigrams.update_layout(yaxis={'categoryorder':'total ascending'})
    fig_bigrams.show()

    # 3. Word Cloud for Visualizing Frequent Terms
    all_text = " ".join(df['cleaned_text'])
    wordcloud = WordCloud(width=1000, height=500, background_color='white', colormap='viridis').generate(all_text)

    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Zudio Discussions')
    plt.show()

# --- Execute EDA ---
if not cleaned_df.empty:
    perform_eda(cleaned_df)


# ==============================================================================
# LAB 5: SOCIAL NETWORK DATA ANALYTICS (INFLUENCER IDENTIFICATION)
# ==============================================================================
# We identify influential posts/topics that drive the most conversation.

def identify_influential_posts(df):
    """
    Identifies and visualizes the most influential posts based on engagement.
    """
    if df.empty:
        print("DataFrame is empty. Skipping Influencer Analysis.")
        return

    print("\n--- Identifying Influential Posts & Topics ---")
    influential_posts = df.sort_values(by='engagement', ascending=False).head(20)

    print("\nTop 20 Most Influential Posts (by Engagement Score):")
    print(influential_posts[['title', 'score', 'num_comments', 'engagement']])

    fig_influencers = px.bar(influential_posts,
                             x='engagement',
                             y='title',
                             orientation='h',
                             color='engagement',
                             color_continuous_scale=px.colors.sequential.Plasma,
                             title='Top 20 Most Engaging Posts')
    fig_influencers.update_layout(yaxis={'categoryorder':'total ascending'}, height=800)
    fig_influencers.show()
    print("\nInsight: These posts highlight the key topics (e.g., protests, product quality, hauls) that capture audience attention the most.")

# --- Execute Influencer Analysis ---
if not cleaned_df.empty:
    identify_influential_posts(cleaned_df)


# ==============================================================================
# LAB 6: CONTENT-BASED ANALYSIS (SENTIMENT, TOPIC, ASPECT)
# ==============================================================================
# The core of the analysis: understanding WHAT people are saying.

def perform_content_analysis(df):
    """
    Performs sentiment, topic, and aspect-based analysis.
    """
    if df.empty:
        print("DataFrame is empty. Skipping Content Analysis.")
        return df

    print("\n--- Starting Content-Based Analysis ---")

    # 1. Sentiment Analysis with VADER
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['full_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    def sentiment_category(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    df['sentiment_category'] = df['sentiment_score'].apply(sentiment_category)

    sentiment_dist = df['sentiment_category'].value_counts()
    fig_sentiment = px.pie(values=sentiment_dist.values,
                           names=sentiment_dist.index,
                           title='Overall Sentiment Distribution of Zudio Discussions',
                           color_discrete_sequence=px.colors.sequential.RdBu)
    fig_sentiment.show()

    # 2. Topic Modeling with LDA
    non_empty_docs = df[df['cleaned_text'].str.len() > 0]['cleaned_text']
    vectorizer = CountVectorizer(max_df=0.9, min_df=10, stop_words='english', ngram_range=(1,2))
    doc_term_matrix = vectorizer.fit_transform(non_empty_docs)

    n_topics = 5
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)

    print("\n--- Identified Key Topics of Discussion ---")
    feature_names = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda.components_):
        top_words = " | ".join([feature_names[i] for i in topic.argsort()[:-11:-1]])
        print(f"Topic #{idx+1}: {top_words}")

    # 3. Aspect-Based Sentiment Analysis (Rule-based approach)
    print("\n--- Performing Aspect-Based Sentiment Analysis ---")
    aspect_keywords = {
        'quality': ['quality', 'fabric', 'material', 'color fade', 'last', 'cheap'],
        'price': ['price', 'affordable', 'cheap', 'cost', 'value', 'expensive', 'rs'],
        'perfume': ['perfume', 'fragrance', 'scent', 'smell', 'lasting', 'bottle'],
        'store_experience': ['store', 'staff', 'customer service', 'billing', 'crowd', 'trial room'],
        'fit_style': ['fit', 'fitting', 'style', 'design', 'crop top', 'jeans', 'dress', 'size']
    }

    # This function will find the sentiment of sentences containing aspect keywords
    def get_aspect_sentiment(text, aspect_keys):
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            if any(key in sentence.lower() for key in aspect_keys):
                return analyzer.polarity_scores(sentence)['compound']
        return np.nan

    for aspect, keywords in aspect_keywords.items():
        df[f'{aspect}_sentiment'] = df['full_text'].apply(lambda text: get_aspect_sentiment(text, keywords))

    # Calculate average sentiment for each aspect
    aspect_sentiment_summary = {aspect: df[f'{aspect}_sentiment'].mean() for aspect in aspect_keywords}
    aspect_df = pd.DataFrame.from_dict(aspect_sentiment_summary, orient='index', columns=['avg_sentiment']).dropna()

    print("\nAverage Sentiment Score per Aspect:")
    print(aspect_df)

    fig_aspect = px.bar(aspect_df,
                        x=aspect_df.index,
                        y='avg_sentiment',
                        color='avg_sentiment',
                        color_continuous_scale='RdYlGn',
                        range_color=[-0.5, 0.5],
                        title='Average Sentiment by Product/Service Aspect')
    fig_aspect.show()
    return df

# --- Execute Content Analysis ---
if not cleaned_df.empty:
    analyzed_df = perform_content_analysis(cleaned_df)
    analyzed_df.to_csv('zudio_full_analysis.csv', index=False)
    print("\nFull analysis saved to 'zudio_full_analysis.csv'")



# ==============================================================================
# LAB 7: DASHBOARD AND REPORTING TOOL
# ==============================================================================
# This part of the code should be saved as a separate file (e.g., dashboard.py)
# and run from the terminal using `streamlit run dashboard.py`

# --- dashboard.py ---
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the fully analyzed data
@st.cache_data
def load_data():
    df = pd.read_csv('zudio_full_analysis.csv')
    df['created_datetime'] = pd.to_datetime(df['created_datetime'])
    return df

df = load_data()

st.set_page_config(layout="wide")
st.title('🛍️ Zudio Social Media Intelligence Dashboard')

# # --- Key Metrics ---
st.header("At a Glance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Posts Analyzed", f"{df.shape[0]}")
col2.metric("Avg. Engagement/Post", f"{int(df['engagement'].mean())}")
positive_posts = df[df['sentiment_category'] == 'Positive'].shape[0]
total_posts = df.shape[0]
col3.metric("Positive Sentiment", f"{(positive_posts/total_posts)*100:.1f}%")
col4.metric("Most Discussed Aspect", "Perfume") # From aspect analysis

# # --- Main Dashboard Layout ---
st.markdown("---")
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Overall Sentiment Distribution")
    sentiment_dist = df['sentiment_category'].value_counts()
    fig_sentiment_pie = px.pie(values=sentiment_dist.values,
                               names=sentiment_dist.index,
                               color_discrete_sequence=['#2ca02c', '#d62728', '#ff7f0e'])
    st.plotly_chart(fig_sentiment_pie, use_container_width=True)

# with right_col:
    st.subheader("Sentiment by Product/Service Aspect")
    aspect_keywords = ['quality', 'price', 'perfume', 'store_experience', 'fit_style']
    aspect_sentiment_summary = {aspect: df[f'{aspect}_sentiment'].mean() for aspect in aspect_keywords}
    aspect_df = pd.DataFrame.from_dict(aspect_sentiment_summary, orient='index', columns=['avg_sentiment']).dropna().reset_index()
    aspect_df.rename(columns={'index': 'Aspect'}, inplace=True)

    fig_aspect_bar = px.bar(aspect_df,
                            x='Aspect',
                            y='avg_sentiment',
                            color='avg_sentiment',
                            color_continuous_scale='RdYlGn_r',
                            range_color=[-0.5, 0.5],
                            title='Average Sentiment per Aspect')
    st.plotly_chart(fig_aspect_bar, use_container_width=True)


st.markdown("---")
st.subheader("Deep Dive into Posts")
sentiment_filter = st.selectbox("Filter posts by sentiment:", ['All'] + list(df['sentiment_category'].unique()))

filtered_df = df
if sentiment_filter != 'All':
    filtered_df = df[df['sentiment_category'] == sentiment_filter]

st.dataframe(filtered_df[['title', 'full_text', 'engagement', 'sentiment_score']].sort_values('engagement', ascending=False).head(10))

# --- End of dashboard.py ---


# ==============================================================================
# LAB 8: DESIGN CREATIVE CONTENT FOR PROMOTION
# ==============================================================================
# Based on our analysis, we propose concrete content strategies.

def generate_content_ideas(df):
    """
    Generates content strategies based on data analysis.
    """
    print("\n--- Actionable Content & Marketing Strategies ---")
    print("\n1. CAMPAIGN: 'The Zudio Scent Experience'")
    print("   - INSIGHT: 'Perfume' is the most positively discussed aspect. Users love the scents and value.")
    print("   - IDEA: Launch a UGC campaign #MyZudioScent. Ask users to post creative photos with their favorite Zudio perfume. Feature the best ones on official channels. Create 'Scent Profile' videos for each perfume, explaining the notes (e.g., 'Madrid: Your burst of citrus for summer days').")

    print("\n2. CAMPAIGN: 'Built to Last, Styled for Now'")
    print("   - INSIGHT: 'Quality' has a mixed-to-negative sentiment. Specific complaints are about color fading and material feel.")
    print("   - IDEA: Create a transparent content series. Show 'Behind the Seams' videos of fabric testing. Post 'Care Guides' on Instagram Stories (e.g., '3 Pro Tips to Keep Your Zudio Jeans Vibrant'). This directly addresses concerns and builds trust.")

    print("\n3. CAMPAIGN: 'Find Your Fit'")
    print("   - INSIGHT: 'Fit & Style' is a key topic. Phrases like 'crop top' and 'jeans fit' are common.")
    print("   - IDEA: Partner with diverse micro-influencers of different body types. Have them create 'Style My Zudio' videos, showing how they style the same item (e.g., a basic white tee) for their body shape. This helps customers visualize the fit better than a standard model.")

# --- Execute Content Idea Generation ---
if not analyzed_df.empty:
    generate_content_ideas(analyzed_df)


# ==============================================================================
# LAB 9: ANALYZE COMPETITOR ACTIVITIES
# ==============================================================================
# A framework for comparing Zudio against its competitors.

def competitor_analysis_framework():
    """
    Outlines the strategy and provides a conceptual framework for competitor analysis.
    """
    print("\n--- Competitor Analysis Framework ---")
    print("STEP 1: Identify Competitors (e.g., Max Fashion, Westside, Yousta).")
    print("STEP 2: Collect Data for each competitor using the same methods (Reddit, Twitter API, etc.).")
    print("STEP 3: Process and Analyze each competitor's data using the same pipeline (Labs 3-6).")
    print("STEP 4: Benchmark and Compare.")

    # Conceptual data for demonstration
    comparison_data = {
        'Brand': ['Zudio', 'Max Fashion', 'Yousta'],
        'Share of Voice (Posts)': [len(analyzed_df), 150, 110],  # Placeholder data
        'Avg. Sentiment': [analyzed_df['sentiment_score'].mean(), 0.15, 0.21], # Placeholder
        'Positive Buzz %': [(analyzed_df.sentiment_category == 'Positive').sum() / len(analyzed_df), 0.65, 0.70], # Placeholder
        'Top Praised Aspect': ['Perfume', 'Kids Wear', 'Ethnic Wear'], # Placeholder
        'Top Complaint': ['Clothing Quality', 'Store Crowd', 'Limited Styles'] # Placeholder
    }
    comparison_df = pd.DataFrame(comparison_data)
    print("\nSample Competitor Benchmark Report:")
    print(comparison_df.to_string())

# --- Execute Competitor Framework ---
competitor_analysis_framework()


# ==============================================================================
# LAB 10: DEVELOP SOCIAL MEDIA ANALYTICS MODELS
# ==============================================================================
# A model to provide actionable inventory recommendations based on social trends.

def inventory_recommendation_model(df):
    """
    A model that identifies trending products and recommends inventory actions,
    weighted by sentiment.
    """
    print("\n--- Trend Detection & Inventory Recommendation Model ---")
    product_keywords = ['perfume', 'jeans', 't-shirt', 'joggers', 'dress', 'footwear', 'shoes', 'kurti']
    recommendations = []

    for product in product_keywords:
        product_df = df[df['full_text'].str.contains(product, case=False, na=False)].copy()
        if len(product_df) < 10:
            continue

        # Calculate a 'trend score' = number of recent mentions * avg sentiment
        # We define 'recent' as the last 30 days from the latest post date
        latest_date = df['created_datetime'].max()
        recent_df = product_df[product_df['created_datetime'] > (latest_date - pd.Timedelta(days=30))]

        if len(recent_df) > 5:
            avg_sentiment = recent_df['sentiment_score'].mean()
            trend_score = len(recent_df) * (1 + avg_sentiment) # Sentiment acts as a multiplier

            if trend_score > 30 and avg_sentiment > 0.1: # High positive buzz
                recommendations.append(
                    f"🚀 STRONG POSITIVE TREND for '{product.title()}'. "
                    f"Recommendation: High priority. Increase stock levels and feature in marketing campaigns."
                )
            elif trend_score > 15: # Moderate buzz
                recommendations.append(
                    f"📈 MODERATE TREND for '{product.title()}'. "
                    f"Recommendation: Monitor sales data closely. Ensure adequate stock."
                )

    if not recommendations:
        print("\nNo strong product trends detected in the last 30 days of data.")
    else:
        print("\nActionable Inventory & Marketing Alerts:")
        for rec in recommendations:
            print(rec)

# --- Execute Analytics Model ---
if not analyzed_df.empty:
    inventory_recommendation_model(analyzed_df)

print("\n\n--- PROJECT EXECUTION COMPLETE ---")
