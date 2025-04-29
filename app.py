# app.py
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
pytrends = TrendReq(hl='en-US', tz=360) # hl=host language, tz=timezone offset (e.g., IST = -330, US East = 360)
# Note: tz=360 is US Eastern Time. For India Standard Time (IST), tz should be -330.
# Using US Eastern for broader Google Trends consistency, but adjust if IST trends are critical.
# pytrends = TrendReq(hl='en-IN', tz=-330) # Example for India

# --- Helper Functions ---

# Cache data for 1 hour (3600 seconds)
# Consider adjusting ttl based on how frequently you expect trends to change significantly
@st.cache_data(ttl=3600)
def get_google_trends(keyword):
    """Fetches Google Trends data: interest over time and related queries."""
    if not keyword:
        return None, None
    try:
        # Build payload for interest over time
        pytrends.build_payload([keyword], cat=0, timeframe='today 3-m', geo='', gprop='') # Last 3 months
        interest_over_time_df = pytrends.interest_over_time()

        # Check if the keyword column exists and the DataFrame is not empty
        if interest_over_time_df.empty or keyword not in interest_over_time_df.columns:
             # No data for the keyword in this timeframe
             st.info(f"No Google Trends interest data found for '{keyword}' in the last 3 months.")
             # Return None to indicate no data, handled downstream
             interest_over_time_df = None # Explicitly set to None
        elif keyword in interest_over_time_df.columns:
             # Ensure 'date' is a column for plotting if data exists
            interest_over_time_df = interest_over_time_df.reset_index()


        # Build payload for related queries (use shorter timeframe)
        pytrends.build_payload([keyword], cat=0, timeframe='today 1-m', geo='', gprop='') # Last 1 month often better
        related_queries_dict = pytrends.related_queries()
        related_queries_df = None # Initialize as None

        # Check if the keyword exists in the dict and has 'top' or 'rising' data
        if keyword in related_queries_dict:
            top_queries = related_queries_dict[keyword].get('top')
            rising_queries = related_queries_dict[keyword].get('rising')

            if top_queries is not None and not top_queries.empty:
                related_queries_df = top_queries
            elif rising_queries is not None and not rising_queries.empty:
                related_queries_df = rising_queries # Fallback to rising

        return interest_over_time_df, related_queries_df

    except requests.exceptions.Timeout:
        st.warning("⏳ Google Trends request timed out. Please try again later.")
        return None, None # Return None, None on timeout
    except Exception as e:
        # Check if the error message contains the 429 code for rate limiting
        if '429' in str(e):
            st.error("🚦 Google Trends Rate Limit Hit (Error 429: Too Many Requests).")
            st.warning("You've made too many requests to Google Trends recently. "
                       "Please wait a while (e.g., 15-60 minutes) before trying again. "
                       "Using the cache for previous results if available.")
        else:
            # General error message for other issues
            st.error(f"An error occurred fetching Google Trends data: {e}")
        # Return None, None on any exception to prevent breaking downstream code
        return None, None


@st.cache_data(ttl=3600) # Cache news data for 1 hour
def get_news_articles(keyword, num_articles=10):
    """Fetches news articles related to the keyword using Google News RSS."""
    if not keyword:
        return []
    try:
        # Using Google News RSS feed for simplicity
        query = keyword.replace(' ', '+')
        # Constructing the Google News RSS feed URL (consider region - US/IN)
        # hl=host language, gl=geographic location
        news_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en" # India-specific
        # news_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en" # US-specific

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
        # Use safety_settings to potentially reduce overly sensitive blocking, if needed
        response = gemini_model.generate_content(
            full_prompt,
            # safety_settings={ # Example: Adjust if needed, be mindful of policy
            #     'HATE': 'BLOCK_ONLY_HIGH',
            #     'HARASSMENT': 'BLOCK_ONLY_HIGH',
            #     'SEXUAL' : 'BLOCK_ONLY_HIGH',
            #     'DANGEROUS' : 'BLOCK_ONLY_HIGH'
            # }
            )

        # Check for valid response text before returning
        # Accessing response.text directly might raise if blocked/no content
        if response.parts:
             return response.text
        elif response.prompt_feedback.block_reason:
             st.warning(f"Gemini content blocked. Reason: {response.prompt_feedback.block_reason}")
             return f"Content blocked by safety filter ({response.prompt_feedback.block_reason})."
        else:
             return "Gemini returned an empty response."

    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        # Check for specific API errors if possible (e.g., content filtering)
        # This part might need adjustment based on the exact exception structure from the google-generativeai library
        feedback = getattr(getattr(e, 'response', None), 'prompt_feedback', None)
        if feedback:
             st.warning(f"Gemini API Feedback: {feedback}")
        return f"Error during analysis: {e}"


# --- Streamlit App Layout ---

st.title("📈 YouTube Trend Hunter")
st.markdown("""
Enter a keyword to explore its trend potential for YouTube content.
This tool fetches data from Google Trends (last 3 months interest, 1 month related queries),
Google News (India), and analyzes it using Google Gemini. Results are cached for 1 hour.
""")
st.caption(f"Current Time: {datetime.now().strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')} (Timezone based on server)")


# --- Input Section ---
keyword = st.text_input("Enter Keyword:", placeholder="e.g., electric vehicles india")
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

        # 3. Prepare text for Gemini analysis
        news_text_for_analysis = ""
        if news_articles:
            news_text_for_analysis = "\n\n".join([
                f"Title: {article['title']}\nSource: {article['link']}\nPublished: {article.get('published', 'N/A')}\nSummary: {article.get('summary', 'No summary available.')}"
                for article in news_articles
            ])
        else:
             news_text_for_analysis = "No relevant news articles found recently for context."
             st.info("📰 No recent news articles found via Google News RSS for detailed analysis.")


        # 4. Gemini Analysis
        gemini_prompts = {
            "summary": "You are a helpful assistant analyzing news for a YouTube creator. Summarize the key themes, topics, and any significant events mentioned in the following news articles related to the keyword. Focus on information relevant for content creation.",
            "sentiment": f"You are a helpful assistant analyzing news for a YouTube creator researching '{keyword}'. Analyze the overall sentiment (Positive, Negative, Neutral) expressed towards '{keyword}' in the provided news text. Provide a one-sentence summary of the sentiment and briefly explain the reasoning based on the articles.",
            "insights": f"You are a helpful assistant analyzing news for a YouTube creator researching '{keyword}'. Based *only* on the provided recent news text, extract 3-5 actionable insights or potential content angles for a YouTube creator focusing on '{keyword}'. Avoid generic advice.",
            "video_ideas": f"You are a helpful YouTube content strategist. Generate 5 distinct video ideas about '{keyword}'. Base these ideas *primarily* on the provided recent news context and any inferred trends. For each idea, provide a catchy Title and a 1-2 sentence Concept description. Format as a numbered list."
        }

        gemini_results = {}

        # Only run news-dependent analysis if news was found
        if news_articles:
            gemini_results["summary"] = analyze_with_gemini(news_text_for_analysis, gemini_prompts["summary"])
            gemini_results["sentiment"] = analyze_with_gemini(news_text_for_analysis, gemini_prompts["sentiment"])
            gemini_results["insights"] = analyze_with_gemini(news_text_for_analysis, gemini_prompts["insights"])
        else:
            gemini_results["summary"] = "Skipped: No news articles found for summarization."
            gemini_results["sentiment"] = "Skipped: No news articles found for sentiment analysis."
            gemini_results["insights"] = "Skipped: No news articles found for insight generation."

        # Video ideas can combine trends (if available) and news (if available)
        trends_context = f"Keyword: {keyword}\n"
        if interest_df is not None and not interest_df.empty:
            trends_context += f"Google Trends shows recent interest (see graph). "
            # You could add more specific trend analysis here if needed (e.g., rising/falling)
        else:
             trends_context += "No specific Google Trends interest data available. "

        if related_queries_df is not None and not related_queries_df.empty:
            related_list = related_queries_df['query'].head().tolist()
            trends_context += f"Related Google searches include: {', '.join(related_list)}."
        else:
             trends_context += "No specific related Google searches found."

        # Combine context for video ideas
        video_ideas_context = f"Context:\n{trends_context}\n\nRecent News Text (if any):\n{news_text_for_analysis if news_articles else 'No recent news articles available.'}"
        gemini_results["video_ideas"] = analyze_with_gemini(video_ideas_context, gemini_prompts["video_ideas"])


    # --- Display Results ---
    st.markdown("---")
    st.subheader("📊 Google Trends Analysis")
    col1, col2 = st.columns([2, 1]) # Make the graph column wider

    with col1:
        st.markdown("**Interest Over Time (Last 3 Months)**")
        if interest_df is not None and not interest_df.empty and keyword in interest_df.columns:
            # Ensure 'date' column is datetime type for Plotly
            interest_df['date'] = pd.to_datetime(interest_df['date'])
            fig = px.line(interest_df, x='date', y=keyword, title=f'Interest in "{keyword}"')
            fig.update_layout(xaxis_title="Date", yaxis_title="Relative Interest Index")
            st.plotly_chart(fig, use_container_width=True)
        elif interest_df is None and '429' not in st.session_state.get('last_error', ''): # Avoid redundant message if rate limited
             # Check if the function returned None due to *no data* vs an error
             st.info(f"Could not retrieve or display Google Trends interest graph for '{keyword}'. This might be due to low search volume or a temporary issue.")
        # Error message for 429 or other errors is handled within get_google_trends function

    with col2:
        st.markdown("**Related Queries (Last 1 Month)**")
        if related_queries_df is not None and not related_queries_df.empty:
            # Display 'Top' or 'Rising' based on what was returned
            st.dataframe(related_queries_df[['query', 'value']], use_container_width=True, hide_index=True)
        else:
            st.info("No related queries found or retrieved.")

    st.markdown("---")

    st.subheader("📰 Recent News & AI Analysis")
    # Use tabs for better organization
    tab_news, tab_analysis = st.tabs(["Recent News Articles", "✨ AI Analysis (Gemini)"])

    with tab_news:
        if news_articles:
            for i, article in enumerate(news_articles):
                st.markdown(f"**{i+1}. {article['title']}**")
                st.markdown(f"<small>Published: {article.get('published', 'N/A')}</small>", unsafe_allow_html=True)
                with st.expander("View Summary & Link"):
                    st.markdown(f"_{article.get('summary', 'No summary available.')}_")
                    st.markdown(f"[Read Full Article]({article['link']})", unsafe_allow_html=True) # Use markdown link
                st.markdown("---") # Separator between articles
        else:
            st.info("No recent news articles found via Google News RSS (India).")

    with tab_analysis:
        st.markdown("**Sentiment Analysis:**")
        st.markdown(gemini_results.get("sentiment", "_Analysis skipped or failed._"))
        st.markdown("**Key Themes Summary:**")
        st.markdown(gemini_results.get("summary", "_Analysis skipped or failed._"))
        st.markdown("**Key Insights/Opportunities:**")
        # Display as list for readability
        insights_text = gemini_results.get("insights", "_Analysis skipped or failed._")
        if insights_text and insights_text not in ["Skipped: No news articles found for insight generation.", "_Analysis skipped or failed._"]:
             # Basic attempt to format if it looks like a list
             if "\n-" in insights_text or "\n*" in insights_text or "\n1." in insights_text:
                 st.markdown(insights_text) # Assume Gemini formatted it okay
             else:
                 # If not formatted as list, just display paragraph
                 st.markdown(f"- {insights_text.replace('.', '.\\n-', 1)}") # Try splitting first sentence
        else:
            st.markdown(insights_text)


    st.markdown("---")
    st.subheader("💡 YouTube Video Ideas (AI Generated)")
    st.markdown(gemini_results.get("video_ideas", "_Could not generate video ideas._"))


elif search_button and not keyword:
    st.warning("Please enter a keyword to analyze.")

# Add footer or credits if desired
st.markdown("---")
st.caption("Powered by Streamlit, Google Trends, Google News RSS, and Google Gemini. Caching enabled for 1 hour.")
# Add a state variable to track errors slightly better if needed
if 'last_error' not in st.session_state:
     st.session_state.last_error = ''
