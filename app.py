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
import io # No need for base64 or os here
import os

# --- Configuration ---
st.set_page_config(
    page_title="Trend Hunter",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Constants ---
DEFAULT_NUM_ARTICLES = 15
CACHE_TTL_SECONDS = 3600 # 1 hour

# --- Font Handling for PDF ---
# (Keep the FONT_PATH logic from the previous version)
FONT_PATH = None
potential_paths = [
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    # Add other potential paths if needed (e.g., for Windows/macOS)
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
    # --- IMPORTANT: Keep the FULL get_google_trends function ---
    # --- from the previous response here. It includes the   ---
    # --- 429 error handling logic.                       ---
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
    # --- Keep the get_news_articles function from the previous response ---
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
        # Removed the st.info from here to avoid duplication when search runs
        return articles
    except Exception as e:
        st.error(f"Error fetching news articles: {e}")
        return []

# Cache Gemini Analysis
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def analyze_with_gemini(text_to_analyze, prompt_instruction):
    # --- Keep the analyze_with_gemini function from the previous response ---
    """Uses Gemini API for analysis."""
    if not text_to_analyze or not GEMINI_API_KEY:
        # Return None or empty string if no input, handle downstream
        return None
    try:
        full_prompt = f"{prompt_instruction}:\n\n---\n{text_to_analyze}\n---"
        response = gemini_model.generate_content(full_prompt)

        if response.parts:
             text_response = response.text.replace('```markdown', '').replace('```', '').strip()
             return text_response
        elif response.prompt_feedback.block_reason:
             st.warning(f"Gemini content blocked. Reason: {response.prompt_feedback.block_reason}")
             return f"Content blocked by safety filter ({response.prompt_feedback.block_reason})."
        else:
             # Explicitly handle empty response case
             return "Gemini returned an empty response. The prompt might require adjustment or the input context was insufficient."

    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        feedback = getattr(getattr(e, 'response', None), 'prompt_feedback', None)
        if feedback:
             st.warning(f"Gemini API Feedback: {feedback}")
        # Return a distinct error message or None
        return f"Error during analysis: {e}"


# --- PDF Generation Function ---
class PDF(FPDF):
    # --- Keep the PDF class definition from the previous response ---
    def header(self):
        if FONT_PATH:
             self.add_font('DejaVu', '', FONT_PATH, uni=True)
             self.set_font('DejaVu', '', 12)
        else:
             self.set_font('Arial', 'B', 12) # Fallback
        self.cell(0, 10, 'Trend Hunter - Research Report', 0, 1, 'C')
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
        if not body: # Handle None or empty strings gracefully
             body = "N/A"
        if FONT_PATH:
            self.set_font('DejaVu', '', 10)
        else:
             self.set_font('Arial', '', 10)
        try:
             safe_body = body.encode('latin-1', 'replace').decode('latin-1')
        except Exception:
             safe_body = str(body) # Ensure it's a string
        self.multi_cell(0, 5, safe_body)
        self.ln()

    def add_link(self, text, link):
        if FONT_PATH:
             self.set_font('DejaVu', '', 10)
        else:
             self.set_font('Arial', '', 10)
        self.set_text_color(0, 0, 255) # Blue for link
        # Ensure text is string
        safe_text = str(text) if text else "Link"
        self.cell(0, 5, safe_text, 0, 1, link=link)
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

    # **MODIFIED**: Changed "YouTube Video Ideas" to "Research Synthesis Report"
    sections = {
        "Sentiment Analysis": gemini_results.get("sentiment"),
        "Key Themes Summary": gemini_results.get("summary"),
        "Key Insights & Opportunities": gemini_results.get("insights"),
        "Research Synthesis Report": gemini_results.get("research_synthesis"), # Use new key
    }

    for title, content in sections.items():
         if FONT_PATH:
            pdf.set_font('DejaVu', '', 12) # Sub-heading font
         else:
            pdf.set_font('Arial', 'B', 12)
         pdf.cell(0, 8, title, 0, 1, 'L')
         pdf.chapter_body(content) # Pass content directly
         pdf.ln(5)


    # --- Google Trends Summary ---
    # (Keep this section as is)
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
    # (Keep this section as is)
    pdf.chapter_title("📰 News Feed (Titles & Links)")
    if news_articles:
         if FONT_PATH:
             pdf.set_font('DejaVu', '', 10)
         else:
              pdf.set_font('Arial', '', 10)
         for i, article in enumerate(news_articles):
              # Ensure title is string before encoding
              safe_title = str(article.get('title', 'N/A'))
              try:
                   encoded_title = safe_title.encode('latin-1', 'replace').decode('latin-1')
              except Exception:
                   encoded_title = safe_title
              pdf.cell(0, 5, f"{i+1}. {encoded_title}", 0, 1)
              pdf.add_link(f"   Read Full Article: {article.get('link', '#')}", article.get('link', '#'))
              # Removed published date and summary from PDF for brevity, kept links
              pdf.ln(3)
    else:
         pdf.chapter_body("No news articles were found or retrieved.")

    # --- Generate PDF Bytes ---
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# --- Streamlit App Layout ---

st.title("📈 Trend Hunter: Research Assistant") # Renamed slightly
st.markdown(f"""
Enter a keyword to research its potential based on recent trends and news.
This tool gathers data from Google Trends, Google News (India, {DEFAULT_NUM_ARTICLES} articles), and uses Google Gemini AI for analysis and synthesis.
**Note on Google Trends Error 429:** This means you've hit Google's request limit. Please **WAIT several hours** before trying again.
""")
current_time_str = datetime.now().strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')
st.caption(f"Current Time (IST): {current_time_str}") # Assuming IST based on location
if not FONT_PATH:
    st.caption("⚠️ Warning: DejaVuSans.ttf font not found. PDF export might have issues with special characters.")


# --- Input Section ---
keyword = st.text_input("Enter Keyword to Research:", placeholder="e.g., drone technology india regulations")
search_button = st.button("🔬 Start Research") # Changed button text

# --- Initialize Session State for results ---
if 'results_data' not in st.session_state:
    st.session_state.results_data = None
if 'last_error' not in st.session_state:
     st.session_state.last_error = ''


# --- Processing Logic ---
if search_button and keyword:
    st.markdown("---")
    st.subheader(f"Research Analysis for: \"{keyword}\"")
    st.session_state.results_data = None # Clear previous results
    st.session_state.last_error = '' # Clear previous error

    with st.spinner("🔬 Conducting research... Fetching data and synthesizing analysis..."):
        # 1. Google Trends
        interest_df, related_queries_df = get_google_trends(keyword)

        # 2. News Articles
        news_articles = get_news_articles(keyword, num_articles=DEFAULT_NUM_ARTICLES)

        # 3. Prepare text for Gemini
        news_text_for_analysis = "No relevant news articles found recently for context." # Default
        if news_articles:
            news_text_for_analysis = "\n\n".join([
                f"Title: {article['title']}\nSource: {article['link']}\nPublished: {article.get('published', 'N/A')}\nSummary: {article.get('summary', 'No summary available.')}"
                for article in news_articles
            ])


        # 4. Gemini Analysis (Modified Prompts - Replaced Video Ideas)
        gemini_prompts = {
             "summary": f"You are a detailed research assistant. Based *only* on the provided news text about '{keyword}', summarize the key themes, topics, significant events, and any mentioned entities or figures. Focus on objective reporting of the news content.",
             "sentiment": f"You are an objective sentiment analyst evaluating news about '{keyword}'. Analyze the overall sentiment (Positive, Negative, Neutral) expressed *in the provided news text*. Provide a one-sentence conclusion and briefly justify it by referencing specific points or tones found in the articles.",
             "insights": f"You are a strategic analyst researching '{keyword}'. Based *only* on the provided news text and trends context, identify 3-5 specific and actionable insights. Consider potential angles, target audience nuances, mentioned challenges or opportunities, and unique perspectives revealed in the news. Frame these as concise bullet points.",
             # *** MODIFIED PROMPT ***
             "research_synthesis": f"You are a research analyst synthesizing information about '{keyword}'. Based *only* on the provided context (which includes recent news summaries, sentiment analysis, key insights, and Google Trends data summary), generate a concise research report section with the following structure:\n\n1.  **Overall Summary:** Briefly combine the key news themes and sentiment.\n2.  **Trend Analysis:** Comment on the implications of the Google Trends data (interest over time and related queries) in relation to the news.\n3.  **Opportunities & Challenges:** Synthesize the key opportunities and potential challenges identified from the insights and news.\n4.  **Concluding Outlook:** Provide a brief concluding thought or outlook based *strictly* on the information provided.\n\nKeep the synthesis objective and directly tied to the input data."
         }

        gemini_results = {}

        # Run analysis only if relevant data exists
        if news_articles:
            gemini_results["summary"] = analyze_with_gemini(news_text_for_analysis, gemini_prompts["summary"])
            gemini_results["sentiment"] = analyze_with_gemini(news_text_for_analysis, gemini_prompts["sentiment"])
            gemini_results["insights"] = analyze_with_gemini(news_text_for_analysis, gemini_prompts["insights"])
        else:
            st.info("📰 No recent news articles found via Google News RSS (India) for detailed analysis.")
            gemini_results["summary"] = "Skipped: No news available."
            gemini_results["sentiment"] = "Skipped: No news available."
            gemini_results["insights"] = "Skipped: No news available."

        # Prepare combined context for the final synthesis step
        trends_context = f"Google Trends Summary:\nKeyword: {keyword}\n"
        if interest_df is not None:
             trends_context += f"- Interest Over Time: Data available (details in graph/report section).\n"
        else:
             trends_context += "- Interest Over Time: Data not retrieved or available.\n"

        if related_queries_df is not None:
             related_list = related_queries_df['query'].head().tolist()
             trends_context += f"- Related Queries Found: {', '.join(related_list)} (and potentially more)."
        else:
             trends_context += "- Related Queries: None found or retrieved."

        # Include prior analysis results in the context for synthesis
        synthesis_context = (
            f"{trends_context}\n\n"
            f"News Summary Provided:\n{gemini_results.get('summary', 'N/A')}\n\n"
            f"Sentiment Analysis Result:\n{gemini_results.get('sentiment', 'N/A')}\n\n"
            f"Identified Insights/Opportunities:\n{gemini_results.get('insights', 'N/A')}\n\n"
            f"Recent News Text Snippets (for additional context):\n{news_text_for_analysis[:1000]}..." # Limit length if needed
        )

        # Run the final synthesis analysis
        gemini_results["research_synthesis"] = analyze_with_gemini(synthesis_context, gemini_prompts["research_synthesis"])

        # Store results in session state
        st.session_state.results_data = {
            "keyword": keyword,
            "interest_df": interest_df,
            "related_queries_df": related_queries_df,
            "news_articles": news_articles,
            "gemini_results": gemini_results
        }


# --- Display Results ---
if st.session_state.results_data:
    results = st.session_state.results_data
    keyword = results["keyword"]
    interest_df = results["interest_df"]
    related_queries_df = results["related_queries_df"]
    news_articles = results["news_articles"]
    gemini_results = results["gemini_results"]

    st.markdown("---")
    st.subheader("📊 Google Trends Analysis")
    # (Keep the Google Trends display columns as before)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Interest Over Time (Last 3 Months)**")
        if interest_df is not None and not interest_df.empty and keyword in interest_df.columns:
            interest_df['date'] = pd.to_datetime(interest_df['date'])
            fig = px.line(interest_df, x='date', y=keyword, title=f'Interest in "{keyword}"')
            fig.update_layout(xaxis_title="Date", yaxis_title="Relative Interest Index")
            st.plotly_chart(fig, use_container_width=True)
        elif st.session_state.last_error != '429':
             st.info(f"Could not retrieve or display Google Trends interest graph for '{keyword}'.")
    with col2:
        st.markdown("**Related Queries (Last 1 Month)**")
        if related_queries_df is not None and not related_queries_df.empty:
            st.dataframe(related_queries_df[['query', 'value']], use_container_width=True, hide_index=True)
        else:
            st.info("No related queries found or retrieved.")


    st.markdown("---")
    st.subheader("📰 News Feed & AI Analysis💡")

    tab_news, tab_analysis = st.tabs(["📰 News Feed", "💡 AI Analysis Details"])

    with tab_news:
        if news_articles:
            st.markdown(f"Displaying the latest {len(news_articles)} articles found via Google News RSS (India):")
            st.markdown("---")
            for i, article in enumerate(news_articles):
                st.markdown(f"**{i+1}. {article.get('title', 'N/A')}**")
                st.markdown(f"<small>Published: {article.get('published', 'N/A')}</small>", unsafe_allow_html=True)
                col_exp, col_link = st.columns([4,1])
                with col_exp:
                     with st.expander("Read Summary"):
                          st.markdown(f"_{article.get('summary', 'No summary available.')}_")
                with col_link:
                     # *** THIS IS THE REDIRECTION LINK ***
                     st.link_button("Read Full Article ↗️", article.get('link', '#'), type="secondary", help="Opens the original article in a new tab")

                st.markdown("---")
        else:
            st.info(f"No news articles found for '{keyword}' to display.")

    with tab_analysis:
        st.markdown("**Sentiment Analysis:**")
        st.markdown(gemini_results.get("sentiment", "_Analysis skipped or failed._"))
        st.markdown("**Key Themes Summary (from News):**")
        st.markdown(gemini_results.get("summary", "_Analysis skipped or failed._"))
        st.markdown("**Key Insights & Opportunities:**")
        st.markdown(gemini_results.get("insights", "_Analysis skipped or failed._"))

    st.markdown("---")
    # *** MODIFIED SECTION ***
    st.subheader(" H1: Research Synthesis Report (AI Generated)")
    st.markdown(gemini_results.get("research_synthesis", "_Could not generate synthesis report._"))

    # --- Download Button ---
    # (Keep this section as is)
    st.markdown("---")
    st.subheader("⬇️ Download Full Report")
    try:
        pdf_data = generate_pdf_report(
            keyword, interest_df, related_queries_df, news_articles, gemini_results
        )
        safe_keyword = "".join(c if c.isalnum() else "_" for c in keyword)
        report_filename = f"Trend_Research_Report_{safe_keyword}.pdf"

        st.download_button(
            label="Download Report as PDF",
            data=pdf_data,
            file_name=report_filename,
            mime="application/pdf",
        )
    except Exception as pdf_error:
         st.error(f"Could not generate PDF report: {pdf_error}")
         st.warning("Ensure 'fpdf2' is installed and fonts (like DejaVu) are accessible if needed.")


elif search_button and not keyword:
    st.warning("Please enter a keyword to research.")
elif st.session_state.last_error == '429':
     # Show persistent error if last known error was 429
     st.error("🚦 Google Trends Rate Limit Hit (Error 429). Please wait several hours before trying again.")


# Footer
st.markdown("---")
st.caption("Powered by Streamlit, Google Trends, Google News RSS, Google Gemini & fpdf2. Caching enabled.")
