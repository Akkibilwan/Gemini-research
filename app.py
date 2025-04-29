import streamlit as st
import pandas as pd
from pytrends.request import TrendReq
import feedparser
import google.generativeai as genai
import plotly.express as px
import requests # Import requests, although feedparser might be sufficient for RSS
from datetime import datetime, timedelta

# --- Configuration ---
st.set_page_config(
    page_title="YouTube Trend Hunter",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed", # Can be "auto", "expanded", "collapsed"
)

# --- API Key and Client Setup ---
try:
    # Load API Key from secrets
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Or another suitable model
except KeyError:
    st.error("🚨 Gemini API Key not found! Please add it to your Streamlit secrets (`.streamlit/secrets.toml`).")
    st.stop()
except Exception as e:
    st.error(f"🚨 Error configuring Gemini: {e}")
    st.stop()

# Pytrends setup
pytrends = TrendReq(hl='en-US', tz=360) # hl=host language, tz=timezone offset (e.g., US East Coast = 360)

# --- Helper Functions ---

@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_google_trends(keyword):
    """Fetches Google Trends data: interest over time and related queries."""
    if not keyword:
        return None, None
    try:
        # Build payload for interest over time
        pytrends.build_payload([keyword], cat=0, timeframe='today 3-m', geo='', gprop='') # Last 3 months
        interest_over_time_df = pytrends.interest_over_time()
        if interest_over_time_df.empty or keyword not in interest_over_time_df.columns:
             # If no data for the keyword, return empty but valid structures
             return pd.DataFrame({'date': [], keyword: []}).set_index('date'), pd.DataFrame()


        # Build payload for related queries (can sometimes fail if keyword is too niche)
        pytrends.build_payload([keyword], cat=0, timeframe='today 1-m', geo='', gprop='') # Shorter timeframe often better for related
        related_queries_dict = pytrends.related_queries()
        related_queries_df = pd.DataFrame()
        if keyword in related_queries_dict and 'top' in related_queries_dict[keyword] and related_queries_dict[keyword]['top'] is not None:
             related_queries_df = related_queries_dict[keyword]['top']
        elif keyword in related_queries_dict and 'rising' in related_queries_dict[keyword] and related_queries_dict[keyword]['rising'] is not None:
             # Fallback to rising if top is empty
             related_queries_df = related_queries_dict[keyword]['rising']


        # Ensure the keyword column exists even if empty before resetting index
        if keyword not in interest_over_time_df.columns:
             interest_over_time_df[keyword] = 0 # Add column of zeros if it doesn't exist

        interest_over_time_df = interest_over_time_df.reset_index() # Make 'date' a column for plotting
        return interest_over_time_df, related_queries_df

    except requests.exceptions.Timeout:
        st.warning("Google Trends request timed out. Try again later.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred fetching Google Trends data: {e}")
        # Return empty dataframes on error to avoid breaking downstream code
        return pd.DataFrame({'date': [], keyword: []}).set_index('date'), pd.DataFrame()


@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_news_articles(keyword, num_articles=10):
    """Fetches news articles related to the keyword using Google News RSS."""
    if not keyword:
        return []
    try:
        # Using Google News RSS feed for simplicity
        # Replace spaces with '+' for the query URL
        query = keyword.replace(' ', '+')
        # Constructing the Google News RSS feed URL
        # Note: This URL structure can change without notice by Google.
        news_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

        feed = feedparser.parse(news_url)

        articles = []
        for entry in feed.entries[:num_articles]:
            articles.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.get("published", "N/A"), # Use .get for safety
                "summary": entry.get("summary", "N/A") # Summary might not always be present
            })
        return articles
    except Exception as e:
        st.error(f"Error fetching news articles: {e}")
        return []

@st.cache_data(ttl=3600) # Cache Gemini results too
def analyze_with_gemini(text_to_analyze, prompt_instruction):
    """Uses Gemini API for analysis (summarization, sentiment, insights, ideas)."""
    if not text_to_analyze or not GEMINI_API_KEY:
        return "No text provided or API key missing."
    try:
        full_prompt = f"{prompt_instruction}:\n\n---\n{text_to_analyze}\n---"
        response = gemini_model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        # Check for specific API errors if possible (e.g., content filtering)
        if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
             st.warning(f"Gemini API Feedback: {e.response.prompt_feedback}")
        return f"Error during analysis: {e}"


# --- Streamlit App Layout ---

st.title("📈 YouTube Trend Hunter")
st.markdown("""
Enter a keyword to explore its trend potential for YouTube content.
This tool fetches data from Google Trends, Google News, and analyzes it using Google Gemini.
""")

# --- Input Section ---
keyword = st.text_input("Enter Keyword:", placeholder="e.g., AI video generation")
search_button = st.button("Analyze Trend ✨")

# --- Results Section ---
if search_button and keyword:
    st.markdown("---") # Separator
    st.subheader(f"Analysis for: \"{keyword}\"")

    with st.spinner("🧙‍♂️ Conjuring insights... Fetching data and analyzing..."):
        # 1. Google Trends
        interest_df, related_queries_df = get_google_trends(keyword)

        # 2. News Articles
        news_articles = get_news_articles(keyword, num_articles=10)

        # Prepare text for Gemini analysis (combine news titles/summaries)
        news_text_for_analysis = "\n".join([
            f"Title: {article['title']}\nSummary: {article.get('summary', 'N/A') or 'No summary available.'}"
            for article in news_articles
        ])
        if not news_text_for_analysis:
             news_text_for_analysis = "No relevant news articles found recently."


        # 3. Gemini Analysis
        gemini_prompts = {
            "summary": "Summarize the key themes and topics from the following news headlines and summaries related to the keyword.",
            "sentiment": f"Analyze the overall sentiment (Positive, Negative, Neutral) towards '{keyword}' based on the following news text. Provide a brief explanation.",
            "insights": f"Based on the recent news and potential trends for '{keyword}', extract 3-5 key insights or potential opportunities for a YouTube creator.",
            "video_ideas": f"Generate 5 diverse YouTube video ideas related to '{keyword}', considering recent news and trends. Format as a numbered list, each with a catchy title and a brief concept description."
        }

        gemini_results = {}
        # Check if there's news text before calling Gemini for analysis based on it
        if news_articles:
            gemini_results["summary"] = analyze_with_gemini(news_text_for_analysis, gemini_prompts["summary"])
            gemini_results["sentiment"] = analyze_with_gemini(news_text_for_analysis, gemini_prompts["sentiment"])
            gemini_results["insights"] = analyze_with_gemini(news_text_for_analysis, gemini_prompts["insights"])
        else:
            # If no news, provide placeholders or perhaps analyze based on trends alone (more complex)
            gemini_results["summary"] = "Insufficient news data for summarization."
            gemini_results["sentiment"] = "Insufficient news data for sentiment analysis."
            gemini_results["insights"] = "Insufficient news data for insight generation."
            st.info("📰 No recent news found for this keyword to fuel deeper analysis.")

        # Video ideas can potentially be generated even without news, based on the keyword itself and trends
        # Combine trends context if available
        trends_context = f"Keyword: {keyword}\n"
        if interest_df is not None and not interest_df.empty:
            trends_context += f"Recent interest shows a trend (see graph). "
        if related_queries_df is not None and not related_queries_df.empty:
            trends_context += f"Related searches include: {', '.join(related_queries_df['query'].head().tolist())}."

        video_ideas_prompt_text = f"{trends_context}\n\n{news_text_for_analysis if news_articles else 'No recent news context available.'}"
        gemini_results["video_ideas"] = analyze_with_gemini(video_ideas_prompt_text, gemini_prompts["video_ideas"])


    # --- Display Results ---
    col1, col2 = st.columns([2, 1]) # Make the graph column wider

    with col1:
        st.subheader("📈 Google Trends: Interest Over Time")
        if interest_df is not None and not interest_df.empty and keyword in interest_df.columns:
            # Ensure 'date' column is datetime type for Plotly
            interest_df['date'] = pd.to_datetime(interest_df['date'])
            fig = px.line(interest_df, x='date', y=keyword, title=f'Interest in "{keyword}" (Last 3 Months)')
            fig.update_layout(xaxis_title="Date", yaxis_title="Relative Interest")
            st.plotly_chart(fig, use_container_width=True)
        elif interest_df is not None and interest_df.empty:
             st.info(f"No Google Trends interest data found for '{keyword}' in the selected timeframe.")
        else:
             st.warning("Could not display Google Trends graph.")


    with col2:
        st.subheader("🔍 Related Queries")
        if related_queries_df is not None and not related_queries_df.empty:
            st.dataframe(related_queries_df, use_container_width=True, hide_index=True)
        else:
            st.info("No related queries found.")

    st.markdown("---")

    st.subheader("📰 Recent News & Analysis")
    news_col, analysis_col = st.columns(2)

    with news_col:
        st.markdown("**Recent News Articles:**")
        if news_articles:
            for i, article in enumerate(news_articles):
                with st.expander(f"{article['title']} ({article.get('published', 'N/A')})"):
                    st.markdown(f"_{article.get('summary', 'No summary available.')}_")
                    st.markdown(f"[Read More]({article['link']})", unsafe_allow_html=True) # Use markdown link
        else:
            st.info("No recent news articles found via Google News RSS.")

    with analysis_col:
        st.markdown("**✨ AI-Powered Analysis (Gemini):**")
        st.markdown("**Sentiment:**")
        st.markdown(gemini_results.get("sentiment", "Analysis not available."))
        st.markdown("**Key Themes Summary:**")
        st.markdown(gemini_results.get("summary", "Analysis not available."))
        st.markdown("**Key Insights/Opportunities:**")
        st.markdown(gemini_results.get("insights", "Analysis not available."))


    st.markdown("---")
    st.subheader("💡 YouTube Video Ideas")
    st.markdown(gemini_results.get("video_ideas", "Could not generate video ideas."))


elif search_button and not keyword:
    st.warning("Please enter a keyword to analyze.")

# Add footer or credits if desired
st.markdown("---")
st.caption("Powered by Streamlit, Google Trends, Google News RSS, and Google Gemini.")
