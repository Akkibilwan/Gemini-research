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
import io
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
FONT_PATH = None
potential_paths = [
    "DejaVuSans.ttf", # Check current directory first
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # Common Linux path
    # Add other potential paths for different OS if needed
]
for path in potential_paths:
    if os.path.exists(path):
        FONT_PATH = path
        break

# --- API Key and Client Setup ---
# Improved API Key Handling
api_key_source = None
try:
    if 'GEMINI_API_KEY' in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        api_key_source = "Streamlit secrets"
    else:
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        if GEMINI_API_KEY:
            api_key_source = "Environment variable"
        else:
             raise ValueError("Gemini API Key not found.") # Raise specific error

    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    # Optionally add a success message or log source
    # st.sidebar.success(f"Gemini API Key loaded from {api_key_source}.")

except ValueError as ve:
     st.error(f"🚨 {ve} Please set it in Streamlit secrets or as an environment variable (GEMINI_API_KEY).")
     st.stop()
except Exception as e:
    st.error(f"🚨 Error configuring Gemini: {e}")
    st.stop()


# Pytrends setup (using India settings)
try:
    pytrends = TrendReq(hl='en-IN', tz=-330) # India settings, IST timezone offset
except Exception as e:
    st.error(f"🚨 Failed to initialize Google Trends connection: {e}")
    st.stop()


# --- Helper Functions ---

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_google_trends(keyword):
    """Fetches Google Trends data with robust error handling."""
    if not keyword: return None, None
    interest_df, related_queries_df = None, None # Initialize
    try:
        pytrends.build_payload([keyword], cat=0, timeframe='today 3-m', geo='', gprop='')
        interest_over_time_df = pytrends.interest_over_time()
        if not interest_over_time_df.empty and keyword in interest_over_time_df.columns:
            interest_df = interest_over_time_df.reset_index()

        # Separate build_payload for related queries if needed, handle potential errors
        try:
            pytrends.build_payload([keyword], cat=0, timeframe='today 1-m', geo='', gprop='')
            related_queries_dict = pytrends.related_queries()
            if keyword in related_queries_dict:
                top_queries = related_queries_dict[keyword].get('top')
                rising_queries = related_queries_dict[keyword].get('rising')
                if top_queries is not None and not top_queries.empty: related_queries_df = top_queries
                elif rising_queries is not None and not rising_queries.empty: related_queries_df = rising_queries
        except Exception as related_err:
            st.warning(f"Could not fetch related queries for '{keyword}': {related_err}")
            # Continue without related queries if this part fails

        return interest_df, related_queries_df

    except requests.exceptions.Timeout:
        st.warning("⏳ Google Trends request timed out.")
        return None, None
    except Exception as e:
        error_str = str(e).lower()
        if '429' in error_str or 'too many requests' in error_str:
            st.error("🚦 Google Trends Rate Limit Hit (Error 429).")
            st.warning("Please WAIT several hours before trying again.")
            st.session_state.last_error = '429'
        elif 'response error' in error_str or 'code 500' in error_str or 'code 400' in error_str:
             st.warning(f"Google Trends returned an error or no data for '{keyword}'. It might be too specific or have low volume.")
             st.session_state.last_error = 'Trends Error'
        else:
            st.error(f"Error fetching Google Trends data: {e}")
            st.session_state.last_error = str(e)
        return None, None


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_news_articles(keyword, num_articles=DEFAULT_NUM_ARTICLES):
    """Fetches and parses news articles using requests and feedparser."""
    if not keyword: return []
    articles = []
    try:
        query = requests.utils.quote(keyword)
        news_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'} # More standard UA
        response = requests.get(news_url, timeout=15, headers=headers) # Increased timeout
        response.raise_for_status()
        feed = feedparser.parse(response.content)

        if feed.bozo:
             exception_details = feed.bozo_exception if hasattr(feed, 'bozo_exception') else 'Unknown parsing issue'
             st.warning(f"Warning: RSS feed parsing issue for '{keyword}'. Results might be incomplete. Details: {exception_details}")

        for entry in feed.entries[:num_articles]:
            articles.append({
                "title": getattr(entry, 'title', 'N/A'),
                "link": getattr(entry, 'link', '#'),
                "published": entry.get("published", "N/A"),
                "summary": getattr(entry, 'summary', None) # Keep as raw object/None initially
            })
        return articles
    except requests.exceptions.Timeout:
         st.warning(f"Timeout fetching news for '{keyword}'.")
         return []
    except requests.exceptions.RequestException as req_e:
         st.error(f"Network error fetching news for '{keyword}': {req_e}")
         return []
    except Exception as e:
        st.error(f"Error processing news feed for '{keyword}': {e}")
        return []


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def analyze_with_gemini(text_to_analyze, prompt_instruction):
    """Uses Gemini API for analysis, with improved error handling."""
    # --- Keep analyze_with_gemini as in previous correct version ---
    if not text_to_analyze or not GEMINI_API_KEY:
        return "Input text missing for analysis."
    try:
        full_prompt = f"{prompt_instruction}\n\nInput Text/Context:\n---\n{text_to_analyze}\n---"
        response = gemini_model.generate_content(full_prompt)
        if response.parts:
             text_response = response.text.replace('```markdown', '').replace('```', '').strip()
             return text_response
        elif response.prompt_feedback.block_reason:
             block_reason = response.prompt_feedback.block_reason
             st.warning(f"Gemini content generation blocked. Reason: {block_reason}")
             return f"Analysis blocked by safety filter (Reason: {block_reason}). Please modify input or prompt."
        else:
             return "Gemini returned an empty response."
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        st.error(f"An error occurred during AI analysis. Please check logs or try again. Error: {e}")
        feedback = getattr(getattr(e, 'response', None), 'prompt_feedback', None)
        if feedback: st.warning(f"Gemini API Feedback on Error: {feedback}")
        return f"Error during AI analysis: {e}"


# --- PDF Generation Function ---
class PDF(FPDF):
    # --- Keep the PDF class definition ---
    def header(self):
        if FONT_PATH:
             self.add_font('DejaVu', '', FONT_PATH, uni=True)
             self.set_font('DejaVu', '', 12)
        else:
             self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Trend Hunter - Research Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        if FONT_PATH: self.set_font('DejaVu', '', 8)
        else: self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        if FONT_PATH: self.set_font('DejaVu', '', 14)
        else: self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        if not body: body = "N/A"
        body_str = str(body)
        if FONT_PATH: self.set_font('DejaVu', '', 10)
        else: self.set_font('Arial', '', 10)
        try: safe_body = body_str.encode('latin-1', 'replace').decode('latin-1')
        except Exception: safe_body = body_str
        self.multi_cell(0, 5, safe_body)
        self.ln()

    def add_link(self, text, link):
        if FONT_PATH: self.set_font('DejaVu', '', 10)
        else: self.set_font('Arial', '', 10)
        self.set_text_color(0, 0, 255)
        safe_text = str(text) if text else "Link"
        self.cell(0, 5, safe_text, 0, 1, link=str(link) if link else '#')
        self.set_text_color(0, 0, 0)
        self.ln(2)


def generate_pdf_report(keyword, interest_df, related_queries_df, news_articles, gemini_results):
    # --- Keep the generate_pdf_report function ---
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_title(f"Research Report for: \"{keyword}\"")
    pdf.chapter_body(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    pdf.ln(10)
    pdf.chapter_title("💡 AI Research & Analysis (Gemini)")
    sections = {
        "Sentiment Analysis": gemini_results.get("sentiment"),
        "Key Themes Summary": gemini_results.get("summary"),
        "Key Insights & Opportunities": gemini_results.get("insights"),
        "Research Synthesis Report": gemini_results.get("research_synthesis"),
    }
    for title, content in sections.items():
         if FONT_PATH: pdf.set_font('DejaVu', '', 12)
         else: pdf.set_font('Arial', 'B', 12)
         pdf.cell(0, 8, title, 0, 1, 'L')
         pdf.chapter_body(content)
         pdf.ln(5)

    pdf.chapter_title("📊 Google Trends Summary")
    if interest_df is not None:
         pdf.chapter_body(f"Interest over time data (last 3 months) analyzed.")
    else:
         pdf.chapter_body(f"Could not retrieve interest over time data.")
    if related_queries_df is not None and not related_queries_df.empty:
         if FONT_PATH: pdf.set_font('DejaVu', '', 12)
         else: pdf.set_font('Arial', 'B', 12)
         pdf.cell(0, 8, "Related Queries (Top/Rising):", 0, 1, 'L')
         if FONT_PATH: pdf.set_font('DejaVu', '', 10)
         else: pdf.set_font('Arial', '', 10)
         for index, row in related_queries_df.head(10).iterrows():
              query_str = str(row.get('query', 'N/A'))
              value_str = str(row.get('value', 'N/A'))
              pdf.cell(0, 5, f"- {query_str} (Value: {value_str})", 0, 1)
         pdf.ln(5)
    else:
         pdf.chapter_body("No related queries data found or retrieved.")

    pdf.chapter_title("📰 News Feed (Titles & Links)")
    if news_articles:
         if FONT_PATH: pdf.set_font('DejaVu', '', 10)
         else: pdf.set_font('Arial', '', 10)
         for i, article in enumerate(news_articles):
              safe_title = str(article.get('title', 'N/A'))
              try: encoded_title = safe_title.encode('latin-1', 'replace').decode('latin-1')
              except Exception: encoded_title = safe_title
              pdf.cell(0, 5, f"{i+1}. {encoded_title}", 0, 1)
              pdf.add_link(f"   Read Full Article", article.get('link', '#'))
              pdf.ln(3)
    else:
         pdf.chapter_body("No news articles were found or retrieved.")

    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# --- Streamlit App Layout ---

st.title("📈 Trend Hunter: Research Assistant")
st.markdown(f"""
Enter a keyword to research its potential based on recent trends and news.
Gathers data from Google Trends, Google News (India, {DEFAULT_NUM_ARTICLES} articles), uses Google Gemini AI for analysis.
**Note on Google Trends Error 429:** This means you've hit Google's request limit. Please **WAIT several hours** before trying again.
""")
current_time_str = datetime.now().strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')
st.caption(f"Current Time: {current_time_str}")
if not FONT_PATH:
    st.caption("⚠️ Warning: DejaVuSans.ttf font not found. PDF export might have issues with special characters.")

# --- Input Section ---
keyword = st.text_input("Enter Keyword to Research:", placeholder="e.g., sustainable tourism kerala", key="keyword_input")
search_button = st.button("🔬 Start Research")

# --- Initialize Session State ---
if 'results_data' not in st.session_state: st.session_state.results_data = None
if 'last_error' not in st.session_state: st.session_state.last_error = ''


# --- Processing Logic ---
if search_button and keyword:
    st.markdown("---")
    st.subheader(f"Research Analysis for: \"{keyword}\"")
    st.session_state.results_data = None
    st.session_state.last_error = ''

    with st.spinner("🔬 Conducting research... Fetching data and synthesizing analysis..."):
        # 1. Google Trends
        interest_df, related_queries_df = get_google_trends(keyword)

        # 2. News Articles
        news_articles = get_news_articles(keyword, num_articles=DEFAULT_NUM_ARTICLES)

        # 3. Prepare text for Gemini
        news_context_for_analysis = "No relevant news articles found recently for detailed analysis."
        num_news = len(news_articles) if news_articles else 0
        if news_articles:
            st.info(f"📰 Found {num_news} news articles. Preparing context for AI analysis...")
            news_context_list = []
            for i, article in enumerate(news_articles):
                raw_summary = article.get('summary') # Get raw value (could be str, None, object)

                # *** CORRECTED BLOCK for context preparation ***
                if isinstance(raw_summary, str):
                    # It's a string, apply cleaning
                    cleaned_summary = raw_summary.replace('<p>', '').replace('</p>', '').replace('<b>', '').replace('</b>', '&nbsp;', ' ')
                else:
                    # It's not a string (e.g., None or other type), use default
                    cleaned_summary = "No summary available."
                # *** END OF CORRECTED BLOCK ***

                # Construct snippet using the safely prepared summary
                snippet = f"Article {i+1} (Title: {article.get('title', 'N/A')} | Published: {article.get('published', 'N/A')}):\nSummary: {cleaned_summary}\nLink: {article.get('link', '#')}\n"
                news_context_list.append(snippet)

            news_context_for_analysis = "\n---\n".join(news_context_list) # Join with separator
        else:
             # Display warning only once if no news found during processing
             st.warning("📰 No news articles found. AI analysis will be based on trends data only (if available).")


        # 4. Gemini Analysis (Prompts emphasizing grounding)
        # --- Keep the gemini_prompts dictionary as defined in the previous correct version ---
        gemini_prompts = {
             "summary": f"You are a factual research assistant. Based *strictly* on the provided news article summaries about '{keyword}', summarize the key factual themes, topics, events, and mentioned entities. Avoid interpretation or adding external information. Focus on accurately reflecting the content of the provided summaries.",
             "sentiment": f"You are an objective sentiment analyst reviewing news about '{keyword}'. Analyze the overall sentiment (Positive, Negative, Neutral) expressed *only within the provided news summaries*. State the dominant sentiment and **justify your conclusion by quoting one or two specific phrases or sentences from the provided text** that strongly support this sentiment rating.",
             "insights": f"You are a strategic analyst researching '{keyword}'. Based *only* on the provided news summaries and trends context, identify 3-5 specific insights or potential opportunities/challenges. **For each insight, explicitly mention which news summary (e.g., 'Article 3 summary mentions...') or trend observation it stems from.** Frame these as concise, evidence-based bullet points.",
             "research_synthesis": f"You are a research analyst synthesizing information about '{keyword}'. Based *strictly and solely* on the provided context below (which includes a Google Trends summary, news article summaries, sentiment analysis, and extracted insights), generate a concise research report section. **Crucially, ground your synthesis in the provided text:** explicitly reference specific news points (e.g., 'As mentioned in Article 2 summary...', 'The trend data suggests X, aligning with the news about Y...'), sentiment findings, or insights when discussing each point. Use direct (but short) quotes from the provided news text *where appropriate* to support your analysis.\n\nStructure the report like this:\n1.  **Overall Summary:** Briefly combine the key news themes and the identified sentiment, referencing specific news points.\n2.  **Trend Connection:** Discuss how the Google Trends data (interest, related queries) relates to or contrasts with the topics found in the news summaries. Cite specific examples.\n3.  **Opportunities & Challenges:** Synthesize the key opportunities and potential challenges identified in the 'insights' section, clearly linking them back to specific news summaries or trend data.\n4.  **Concluding Outlook:** Provide a brief concluding thought based *only* on the synthesized information, reiterating the main evidence."
        }

        gemini_results = {}
        analysis_skipped_message = "Analysis skipped as no news articles were found."

        # Initialize results
        gemini_results["summary"] = analysis_skipped_message
        gemini_results["sentiment"] = analysis_skipped_message
        gemini_results["insights"] = analysis_skipped_message
        gemini_results["research_synthesis"] = "Synthesis skipped or incomplete due to missing news data."

        # Run analysis depending on news context
        if news_articles:
            gemini_results["summary"] = analyze_with_gemini(news_context_for_analysis, gemini_prompts["summary"]) or "AI analysis failed for Summary."
            gemini_results["sentiment"] = analyze_with_gemini(news_context_for_analysis, gemini_prompts["sentiment"]) or "AI analysis failed for Sentiment."
            gemini_results["insights"] = analyze_with_gemini(news_context_for_analysis, gemini_prompts["insights"]) or "AI analysis failed for Insights."

        # Prepare combined context for synthesis
        trends_context = f"Google Trends Context:\nKeyword: {keyword}\n"
        trends_context += "- Interest Over Time: Data available.\n" if interest_df is not None else "- Interest Over Time: Data not retrieved or available.\n"
        if related_queries_df is not None:
             related_list = related_queries_df['query'].head().tolist()
             trends_context += f"- Related Queries Found: {', '.join(map(str, related_list))}."
        else: trends_context += "- Related Queries: None found or retrieved."

        synthesis_input_context = (
            f"{trends_context}\n\n"
            f"Analysis based on News Summaries (if available):\n"
            f"News Themes Summary:\n{gemini_results.get('summary', 'N/A')}\n\n"
            f"Sentiment Analysis Result:\n{gemini_results.get('sentiment', 'N/A')}\n\n"
            f"Identified Insights/Opportunities:\n{gemini_results.get('insights', 'N/A')}\n\n"
            f"Source News Summaries Used (max {num_news}):\n---\n{news_context_for_analysis}\n---"
        )

        # Run synthesis
        gemini_results["research_synthesis"] = analyze_with_gemini(synthesis_input_context, gemini_prompts["research_synthesis"]) or "AI analysis failed for Synthesis."

        # Store results
        st.session_state.results_data = {
            "keyword": keyword, "interest_df": interest_df, "related_queries_df": related_queries_df,
            "news_articles": news_articles, "gemini_results": gemini_results
        }


# --- Display Results ---
if st.session_state.results_data:
    # --- Keep the Display logic sections ---
    # --- It will now display the output from the revised prompts ---
    # --- Includes the fix for displaying summaries using isinstance() ---
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
            try:
                interest_df['date'] = pd.to_datetime(interest_df['date'])
                fig = px.line(interest_df, x='date', y=keyword, title=f'Interest in "{keyword}"')
                fig.update_layout(xaxis_title="Date", yaxis_title="Relative Interest Index")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as plot_err:
                st.warning(f"Could not plot Google Trends graph: {plot_err}")
        elif interest_df is None and st.session_state.last_error != '429':
             st.info(f"Could not retrieve Google Trends interest data for '{keyword}'.")
    with col2:
        st.markdown("**Related Queries (Last 1 Month)**")
        if related_queries_df is not None and not related_queries_df.empty:
            try:
                st.dataframe(related_queries_df[['query', 'value']], use_container_width=True, hide_index=True)
            except Exception as df_err:
                st.warning(f"Could not display related queries: {df_err}")
        else:
            st.info("No related queries found or retrieved.")

    st.markdown("---")
    st.subheader("📰 News Feed & AI Analysis💡")
    tab_news, tab_analysis = st.tabs(["📰 News Feed", "💡 AI Analysis Details"])

    with tab_news:
        if news_articles:
            st.markdown(f"Displaying {len(news_articles)} articles found via Google News RSS (India):")
            st.markdown("---")
            for i, article in enumerate(news_articles):
                st.markdown(f"**{i+1}. {article.get('title', 'N/A')}**")
                st.markdown(f"<small>Published: {article.get('published', 'N/A')}</small>", unsafe_allow_html=True)
                col_exp, col_link = st.columns([4,1])
                with col_exp:
                     # Using the corrected block with isinstance() check
                     raw_summary = article.get('summary')
                     if isinstance(raw_summary, str):
                         cleaned_summary = raw_summary.replace('<p>', '').replace('</p>', '').replace('<b>', '').replace('</b>', '&nbsp;', ' ')
                     else:
                         cleaned_summary = "No summary available."

                     with st.expander("Read Summary"):
                          st.markdown(f"_{cleaned_summary}_", unsafe_allow_html=True)
                with col_link:
                     link_url = article.get('link', '#')
                     st.link_button("Read Full Article ↗️", link_url, type="secondary", help="Opens the original article in a new tab")
                st.markdown('<hr style="margin-top:0.5rem; margin-bottom:0.5rem;">', unsafe_allow_html=True)
        else:
            # Display message if search was attempted but no articles found
            if search_button:
                st.info(f"No news articles found for '{keyword}' via Google News RSS (India).")

    with tab_analysis:
        st.markdown("**Sentiment Analysis (Based on News Summaries):**")
        st.markdown(gemini_results.get("sentiment", "_Analysis skipped or failed._"))
        st.markdown("**Key Themes Summary (Extracted from News Summaries):**")
        st.markdown(gemini_results.get("summary", "_Analysis skipped or failed._"))
        st.markdown("**Key Insights & Opportunities (Derived from News/Trends):**")
        st.markdown(gemini_results.get("insights", "_Analysis skipped or failed._"))
        st.caption("Note: Analysis attempts to link findings to specific news items or trends as requested.")

    st.markdown("---")
    st.subheader(" H1: Research Synthesis Report (AI Generated)")
    st.markdown(gemini_results.get("research_synthesis", "_Could not generate synthesis report._"))
    st.caption("Note: Synthesis attempts to ground findings in the provided news/trends data and cite specifics as requested.")


    # --- Download Button ---
    st.markdown("---")
    st.subheader("⬇️ Download Full Report")
    if gemini_results or news_articles or interest_df is not None or related_queries_df is not None:
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
                key="pdf_download_button"
            )
        except Exception as pdf_error:
             st.error(f"Could not generate PDF report: {pdf_error}")
             st.warning("Ensure 'fpdf2' is installed and fonts (like DejaVu) are accessible.")
    else:
         st.info("No data available to generate a PDF report.")


# --- Handle cases where search button wasn't pressed or keyword is empty ---
elif search_button and not keyword:
    st.warning("Please enter a keyword to research.")

# --- Display persistent error message if rate limit was hit last time ---
elif 'last_error' in st.session_state and st.session_state.last_error == '429':
     st.error("🚦 Google Trends Rate Limit Hit previously. Please wait several hours before trying again.")

# Footer
st.markdown("---")
st.caption("Powered by Streamlit, Google Trends, Google News RSS, Google Gemini & fpdf2. Caching enabled.")
