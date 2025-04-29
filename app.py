# app.py
import streamlit as st
import pandas as pd
from pytrends.request import TrendReq
import feedparser
import google.generativeai as genai
import plotly.express as px
import requests
from datetime import datetime
# Added for PDF Generation
from fpdf import FPDF
import base64
import io
import os

# --- Configuration ---
st.set_page_config(
    page_title="YouTube Trend Hunter",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Constants ---
DEFAULT_NUM_ARTICLES = 15 # Increased for potentially "deeper" analysis
CACHE_TTL_SECONDS = 3600 # 1 hour

# --- Font Handling for PDF ---
# Check if a common font file exists, otherwise fall back.
# You might need to install DejaVu fonts on your system or place the .ttf file
# in the same directory as the script for reliable Unicode support in PDFs.
# Example: sudo apt-get install fonts-dejavu-core (on Debian/Ubuntu)
FONT_PATH = None
potential_paths = [
    "DejaVuSans.ttf", # Check current dir
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # Common Linux path
]
for path in potential_paths:
    if os.path.exists(path):
        FONT_PATH = path
        break

# --- API Key and Client Setup ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except KeyError:
    st.error("🚨 Gemini API Key not found! Add it to `.streamlit/secrets.toml`.")
    st.stop()
except Exception as e:
    st.error(f"🚨 Error configuring Gemini: {e}")
    st.stop()

# Pytrends setup (using India settings)
pytrends = TrendReq(hl='en-IN', tz=-330) # India settings

# --- Helper Functions ---

# Cache Google Trends data
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_google_trends(keyword):
    # ... (Keep the previously updated get_google_trends function here) ...
    # (Including the check for 429 errors and returning None, None on failure)
    """Fetches Google Trends data: interest over time and related queries."""
    if not keyword:
        return None, None
    try:
        # Build payload for interest over time
        pytrends.build_payload([keyword], cat=0, timeframe='today 3-m', geo='', gprop='') # Last 3 months
        interest_over_time_df = pytrends.interest_over_time()

        # Check if the keyword column exists and the DataFrame is not empty
        if interest_over_time_df.empty or keyword not in interest_over_time_df.columns:
             st.info(f"No Google Trends interest data found for '{keyword}' in the last 3 months.")
             interest_over_time_df = None # Explicitly set to None
        elif keyword in interest_over_time_df.columns:
            interest_over_time_df = interest_over_time_df.reset_index()

        # Build payload for related queries (use shorter timeframe)
        pytrends.build_payload([keyword], cat=0, timeframe='today 1-m', geo='', gprop='') # Last 1 month
        related_queries_dict = pytrends.related_queries()
        related_queries_df = None # Initialize as None

        if keyword in related_queries_dict:
            top_queries = related_queries_dict[keyword].get('top')
            rising_queries = related_queries_dict[keyword].get('rising')
            if top_queries is not None and not top_queries.empty:
                related_queries_df = top_queries
            elif rising_queries is not None and not rising_queries.empty:
                related_queries_df = rising_queries

        return interest_over_time_df, related_queries_df

    except requests.exceptions.Timeout:
        st.warning("⏳ Google Trends request timed out. Please try again later.")
        return None, None
    except Exception as e:
        if '429' in str(e):
            st.error("🚦 Google Trends Rate Limit Hit (Error 429).")
            st.warning("Too many requests sent to Google Trends. Please WAIT several hours before trying again. Using cache if available.")
            st.session_state.last_error = '429' # Track the error
        else:
            st.error(f"An error occurred fetching Google Trends data: {e}")
            st.session_state.last_error = str(e)
        return None, None

# Cache News Articles
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_news_articles(keyword, num_articles=DEFAULT_NUM_ARTICLES):
    """Fetches news articles related to the keyword using Google News RSS (India)."""
    if not keyword:
        return []
    try:
        query = keyword.replace(' ', '+')
        news_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en" # India-specific
        feed = feedparser.parse(news_url)
        articles = []
        for entry in feed.entries[:num_articles]:
            articles.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.get("published", "N/A"),
                "summary": entry.get("summary", "N/A")
            })
        if not articles:
             st.info(f"📰 No recent news articles found via Google News RSS for '{keyword}' (India).")
        return articles
    except Exception as e:
        st.error(f"Error fetching news articles: {e}")
        return []

# Cache Gemini Analysis
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def analyze_with_gemini(text_to_analyze, prompt_instruction):
    """Uses Gemini API for analysis."""
    # ... (Keep the previously updated analyze_with_gemini function here) ...
    if not text_to_analyze or not GEMINI_API_KEY:
        return "No text provided or API key missing."
    try:
        full_prompt = f"{prompt_instruction}:\n\n---\n{text_to_analyze}\n---"
        response = gemini_model.generate_content(full_prompt)

        if response.parts:
             # Clean up potential markdown issues before returning
             text_response = response.text.replace('```markdown', '').replace('```', '').strip()
             return text_response
        elif response.prompt_feedback.block_reason:
             st.warning(f"Gemini content blocked. Reason: {response.prompt_feedback.block_reason}")
             return f"Content blocked by safety filter ({response.prompt_feedback.block_reason})."
        else:
             return "Gemini returned an empty response."

    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        feedback = getattr(getattr(e, 'response', None), 'prompt_feedback', None)
        if feedback:
             st.warning(f"Gemini API Feedback: {feedback}")
        return f"Error during analysis: {e}"

# --- PDF Generation Function ---
class PDF(FPDF):
    def header(self):
        if FONT_PATH:
             self.add_font('DejaVu', '', FONT_PATH, uni=True)
             self.set_font('DejaVu', '', 12)
        else:
             self.set_font('Arial', 'B', 12) # Fallback
        self.cell(0, 10, 'YouTube Trend Hunter - Research Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        if FONT_PATH:
            self.set_font('DejaVu', '', 8)
        else:
             self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        if FONT_PATH:
            self.set_font('DejaVu', '', 14)
        else:
             self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        if FONT_PATH:
            self.set_font('DejaVu', '', 10)
        else:
             self.set_font('Arial', '', 10)
        # Use multi_cell to handle automatic line breaks and Unicode
        # Decode explicitly to handle potential encoding issues from Gemini
        try:
             # Attempt to encode/decode to handle potential weird characters gracefully
             safe_body = body.encode('latin-1', 'replace').decode('latin-1')
        except Exception:
             safe_body = body # Fallback if encoding fails
        self.multi_cell(0, 5, safe_body)
        self.ln()

    def add_link(self, text, link):
        if FONT_PATH:
             self.set_font('DejaVu', '', 10)
        else:
             self.set_font('Arial', '', 10)
        self.set_text_color(0, 0, 255) # Blue for link
        self.cell(0, 5, text, 0, 1, link=link)
        self.set_text_color(0, 0, 0) # Reset color
        self.ln(2)


def generate_pdf_report(keyword, interest_df, related_queries_df, news_articles, gemini_results):
    """Generates a PDF report from the collected data."""
    pdf = PDF()
    pdf.add_page()

    # --- Report Header ---
    pdf.chapter_title(f"Research Report for: \"{keyword}\"")
    pdf.chapter_body(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    pdf.ln(10)

    # --- Gemini Analysis Sections ---
    pdf.chapter_title("💡 AI Research & Analysis (Gemini)")

    sections = {
        "Sentiment Analysis": gemini_results.get("sentiment", "N/A"),
        "Key Themes Summary": gemini_results.get("summary", "N/A"),
        "Key Insights & Opportunities": gemini_results.get("insights", "N/A"),
        "YouTube Video Ideas": gemini_results.get("video_ideas", "N/A"),
    }

    for title, content in sections.items():
         if FONT_PATH:
            pdf.set_font('DejaVu', '', 12) # Sub-heading font
         else:
            pdf.set_font('Arial', 'B', 12)
         pdf.cell(0, 8, title, 0, 1, 'L')
         pdf.chapter_body(content if content else "N/A") # Ensure content is not None
         pdf.ln(5)


    # --- Google Trends (Optional Text Summary) ---
    # Including the graph directly is complex, so provide text summary
    pdf.chapter_title("📊 Google Trends Summary")
    if interest_df is not None:
         pdf.chapter_body(f"Interest over time data (last 3 months) was analyzed for '{keyword}'. Refer to the web app for the visual graph.")
    else:
         pdf.chapter_body(f"Could not retrieve interest over time data for '{keyword}'.")

    if related_queries_df is not None and not related_queries_df.empty:
         if FONT_PATH:
            pdf.set_font('DejaVu', '', 12)
         else:
             pdf.set_font('Arial', 'B', 12)
         pdf.cell(0, 8, "Related Queries (Top/Rising):", 0, 1, 'L')
         if FONT_PATH:
             pdf.set_font('DejaVu', '', 10)
         else:
              pdf.set_font('Arial', '', 10)
         for index, row in related_queries_df.head(10).iterrows(): # Show top 10
              pdf.cell(0, 5, f"- {row['query']} (Value: {row['value']})", 0, 1)
         pdf.ln(5)
    else:
         pdf.chapter_body("No related queries data found or retrieved.")


    # --- News Feed Section ---
    pdf.chapter_title("📰 News Feed")
    if news_articles:
         if FONT_PATH:
             pdf.set_font('DejaVu', '', 10)
         else:
              pdf.set_font('Arial', '', 10)
         for i, article in enumerate(news_articles):
              pdf.cell(0, 5, f"{i+1}. {article['title']}", 0, 1)
              pdf.add_link(f"   Read Full Article: {article['link']}", article['link'])
              pdf.chapter_body(f"   Published: {article.get('published', 'N/A')}")
              # pdf.chapter_body(f"   Summary: {article.get('summary', 'N/A')}") # Optional: Include summary
              pdf.ln(3)
    else:
         pdf.chapter_body("No news articles were found or retrieved.")

    # --- Generate PDF Bytes ---
    # Use BytesIO to avoid saving to disk on the server
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# --- Streamlit App Layout ---

st.title("📈 YouTube Trend Hunter")
st.markdown(f"""
Enter a keyword to explore its trend potential. Fetches data from Google Trends, Google News (India, {DEFAULT_NUM_ARTICLES} articles), and analyzes using Gemini.
**Note on Google Trends Error 429:** This error means you've hit Google's request limit. Please wait several hours before trying again. The app uses caching, but new keywords still require requests.
""")
# Display current time and info about font for PDF
current_time_str = datetime.now().strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')
st.caption(f"Current Time: {current_time_str}")
if not FONT_PATH:
    st.caption("⚠️ Warning: DejaVuSans.ttf font not found. PDF export might have issues with special characters.")
else:
    st.caption(f"✔️ Using font: {FONT_PATH} for PDF export.")


# --- Input Section ---
keyword = st.text_input("Enter Keyword:", placeholder="e.g., sustainable fashion india")
search_button = st.button("Analyze Trend ✨")

# --- Initialize Session State for results ---
if 'results_data' not in st.session_state:
    st.session_state.results_data = None
if 'last_error' not in st.session_state:
     st.session_state.last_error = ''


# --- Processing Logic ---
if search_button and keyword:
    st.markdown("---")
    st.subheader(f"Analysis for: \"{keyword}\"")
    st.session_state.results_data = None # Clear previous results before new search

    with st.spinner("🧙‍♂️ Conjuring insights... This may take a moment..."):
        # 1. Google Trends
        interest_df, related_queries_df = get_google_trends(keyword)

        # 2. News Articles
        news_articles = get_news_articles(keyword, num_articles=DEFAULT_NUM_ARTICLES)

        # 3. Prepare text for Gemini
        news_text_for_analysis = ""
        if news_articles:
            news_text_for_analysis = "\n\n".join([
                f"Title: {article['title']}\nSource: {article['link']}\nPublished: {article.get('published', 'N/A')}\nSummary: {article.get('summary', 'No summary available.')}"
                for article in news_articles
            ])
        else:
             news_text_for_analysis = "No relevant news articles found recently for context."


        # 4. Gemini Analysis (Refined Prompts)
        gemini_prompts = {
             "summary": f"You are a detailed research assistant. Based *only* on the provided news text about '{keyword}', summarize the key themes, topics, significant events, and any mentioned entities or figures. Highlight information most relevant for understanding the current landscape.",
             "sentiment": f"You are an objective sentiment analyst evaluating news about '{keyword}'. Analyze the overall sentiment (Positive, Negative, Neutral) expressed *in the provided news text*. Provide a one-sentence conclusion and briefly justify it by referencing specific points or tones found in the articles.",
             "insights": f"You are a strategic analyst for a YouTube creator researching '{keyword}'. Based *only* on the provided news text and trends context, identify 3-5 specific and actionable insights. Consider potential content angles, target audience nuances, mentioned challenges or opportunities, and unique perspectives revealed in the news.",
             "video_ideas": f"You are an innovative YouTube content strategist brainstorming ideas about '{keyword}'. Generate 5 distinct and engaging video ideas. Base these ideas *primarily* on the provided recent news context and trends summary. For each idea, provide a Click-worthy Title and a 2-3 sentence Concept outlining the video's hook, core content, and potential value to the viewer. Format as a numbered list."
         }

        gemini_results = {}

        if news_articles:
            gemini_results["summary"] = analyze_with_gemini(news_text_for_analysis, gemini_prompts["summary"])
            gemini_results["sentiment"] = analyze_with_gemini(news_text_for_analysis, gemini_prompts["sentiment"])
            gemini_results["insights"] = analyze_with_gemini(news_text_for_analysis, gemini_prompts["insights"])
        else:
            gemini_results["summary"] = "Skipped: No news articles for summarization."
            gemini_results["sentiment"] = "Skipped: No news articles for sentiment analysis."
            gemini_results["insights"] = "Skipped: No news articles for insight generation."

        trends_context = f"Keyword: {keyword}\n"
        if interest_df is not None:
             trends_context += f"Google Trends shows recent interest (see graph if available). "
        else:
             trends_context += "Google Trends interest data was not retrieved. "

        if related_queries_df is not None:
             related_list = related_queries_df['query'].head().tolist()
             trends_context += f"Related Google searches include: {', '.join(related_list)}."
        else:
             trends_context += "No related Google searches found or retrieved."

        video_ideas_context = f"Context:\n{trends_context}\n\nRecent News Text (if any):\n{news_text_for_analysis}"
        gemini_results["video_ideas"] = analyze_with_gemini(video_ideas_context, gemini_prompts["video_ideas"])

        # Store results in session state for download button
        st.session_state.results_data = {
            "keyword": keyword,
            "interest_df": interest_df,
            "related_queries_df": related_queries_df,
            "news_articles": news_articles,
            "gemini_results": gemini_results
        }


# --- Display Results ---
# Check if results exist in session state before trying to display
if st.session_state.results_data:
    results = st.session_state.results_data
    keyword = results["keyword"]
    interest_df = results["interest_df"]
    related_queries_df = results["related_queries_df"]
    news_articles = results["news_articles"]
    gemini_results = results["gemini_results"]

    st.markdown("---")
    st.subheader("📊 Google Trends Analysis")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Interest Over Time (Last 3 Months)**")
        if interest_df is not None and not interest_df.empty and keyword in interest_df.columns:
            interest_df['date'] = pd.to_datetime(interest_df['date'])
            fig = px.line(interest_df, x='date', y=keyword, title=f'Interest in "{keyword}"')
            fig.update_layout(xaxis_title="Date", yaxis_title="Relative Interest Index")
            st.plotly_chart(fig, use_container_width=True)
        elif st.session_state.last_error != '429': # Only show generic message if not rate limited
             st.info(f"Could not retrieve or display Google Trends interest graph for '{keyword}'.")

    with col2:
        st.markdown("**Related Queries (Last 1 Month)**")
        if related_queries_df is not None and not related_queries_df.empty:
            st.dataframe(related_queries_df[['query', 'value']], use_container_width=True, hide_index=True)
        else:
            st.info("No related queries found or retrieved.")

    st.markdown("---")
    st.subheader("📰 News Feed & AI Analysis💡")

    tab_news, tab_analysis = st.tabs(["📰 News Feed", "💡 AI Research & Analysis"])

    with tab_news:
        if news_articles:
            st.markdown(f"Displaying the latest {len(news_articles)} articles found via Google News RSS (India):")
            st.markdown("---")
            for i, article in enumerate(news_articles):
                st.markdown(f"**{i+1}. {article['title']}**")
                st.markdown(f"<small>Published: {article.get('published', 'N/A')}</small>", unsafe_allow_html=True)
                # Use columns for better layout of summary and link
                col_exp, col_link = st.columns([4,1])
                with col_exp:
                     with st.expander("Read Summary"):
                          st.markdown(f"_{article.get('summary', 'No summary available.')}_")
                with col_link:
                     st.link_button("Read Full Article ↗️", article['link'], type="secondary")

                st.markdown("---") # Separator between articles
        else:
            st.info(f"No news articles found for '{keyword}' to display.")

    with tab_analysis:
        st.markdown("**Sentiment Analysis:**")
        st.markdown(gemini_results.get("sentiment", "_Analysis skipped or failed._"))
        st.markdown("**Key Themes Summary:**")
        st.markdown(gemini_results.get("summary", "_Analysis skipped or failed._"))
        st.markdown("**Key Insights & Opportunities:**")
        st.markdown(gemini_results.get("insights", "_Analysis skipped or failed._"))

    st.markdown("---")
    st.subheader("💡 YouTube Video Ideas (AI Generated)")
    st.markdown(gemini_results.get("video_ideas", "_Could not generate video ideas._"))

    # --- Download Button ---
    st.markdown("---")
    st.subheader("⬇️ Download Full Report")
    try:
        pdf_data = generate_pdf_report(
            keyword, interest_df, related_queries_df, news_articles, gemini_results
        )
        # Sanitize keyword for filename
        safe_keyword = "".join(c if c.isalnum() else "_" for c in keyword)
        report_filename = f"YouTube_Trend_Report_{safe_keyword}.pdf"

        st.download_button(
            label="Download Report as PDF",
            data=pdf_data,
            file_name=report_filename,
            mime="application/pdf",
        )
    except Exception as pdf_error:
         st.error(f"Could not generate PDF report: {pdf_error}")
         st.warning("Ensure the 'fpdf2' library is installed and fonts (like DejaVu) are accessible if needed for special characters.")


elif search_button and not keyword:
    st.warning("Please enter a keyword to analyze.")
elif st.session_state.last_error == '429':
     st.error("🚦 Google Trends Rate Limit Hit (Error 429). Please wait several hours before trying again.")


# Footer
st.markdown("---")
st.caption("Powered by Streamlit, Google Trends, Google News RSS, Google Gemini & fpdf2. Caching enabled.")
