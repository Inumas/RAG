import streamlit as st
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from ingestion import load_data
from database import index_documents, clear_database
from rag_engine import query_rag
from retrieval import HybridRetriever

@st.cache_resource
def get_retriever():
    return HybridRetriever()

st.set_page_config(page_title="The Batch RAG", layout="wide")

st.title("🤖 The Batch - Multimodal RAG System")

# Sidebar for Setup
with st.sidebar:
    st.header("Setup")
    # Try to get from env first
    env_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API Key", value=env_key, type="password")
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.divider()
    
    # Database status indicator
    try:
        retriever = get_retriever()
        doc_count = len(retriever.bm25_docs)
        if doc_count > 0:
            st.success(f"📚 Database: {doc_count} documents indexed")
        else:
            st.warning("📭 Database: Empty - Please ingest data first")
    except Exception:
        st.info("📭 Database: Not initialized")
    
    st.divider()
    
    ingestion_mode = st.radio(
        "Ingestion Mode", 
        ["📥 Add Missing (100)", "🗂️ Full Sitemap (1000)", "📰 Issue Range", "🔗 Single URL"]
    )
    
    start_issue = 330
    end_issue = 334
    max_articles = 100
    custom_url = ""
    
    if ingestion_mode == "📰 Issue Range":
        c1, c2 = st.columns(2)
        start_issue = c1.number_input("Start Issue", min_value=1, value=330)
        end_issue = c2.number_input("End Issue", min_value=1, value=334)
    elif ingestion_mode == "🗂️ Full Sitemap (1000)":
        max_articles = st.slider("Max articles to ingest", 100, 2000, 1000)
        st.info(f"⏱️ Estimated time: ~{max_articles * 2 // 60} minutes")
    elif ingestion_mode == "🔗 Single URL":
        custom_url = st.text_input("Article URL", placeholder="https://www.deeplearning.ai/the-batch/...")
    
    clear_db = st.checkbox("Clear existing database first?", value=False)
    
    if st.button("🔄 Ingest Data"):
        if not api_key:
            st.error("Please enter an OpenAI API Key first.")
        else:
            with st.status("Ingesting data...", expanded=True) as status:
                if clear_db:
                    st.write("Clearing old database...")
                    clear_database()
                
                docs = []
                
                if ingestion_mode == "📰 Issue Range":
                    st.write(f"Scraping Issues {start_issue} to {end_issue}...")
                    docs = load_data(mode="issues", start_issue=start_issue, end_issue=end_issue)
                
                elif ingestion_mode == "🗂️ Full Sitemap (1000)":
                    st.write(f"Fetching {max_articles} most recent articles from sitemap...")
                    docs = load_data(mode="sitemap", max_articles=max_articles, skip_existing=True)
                
                elif ingestion_mode == "📥 Add Missing (100)":
                    st.write("Finding and ingesting 100 new articles not in database...")
                    docs = load_data(mode="sitemap", max_articles=100, skip_existing=True)
                
                elif ingestion_mode == "🔗 Single URL" and custom_url:
                    st.write(f"Scraping {custom_url}...")
                    from ingestion import scrape_article
                    docs = scrape_article(custom_url) or []
                
                st.write(f"Scraped {len(docs)} documents.")
                
                if docs:
                    st.write("Indexing into Vector DB...")
                    index_documents(docs)
                else:
                    st.warning("No new documents found.")
                
                status.update(label="Ingestion Complete!", state="complete", expanded=False)
            
            # Clear cache to force retriever reload with new data
            st.cache_resource.clear()
            st.success("Data ready! Refreshing...")
            st.rerun()

    st.divider()
    st.header("🔍 Data Filters")
    
    # Topic Filter
    topics = ["All", "Generative AI", "LLMs", "Computer Vision", "Robotics", "AI Ethics", "Hardware", "AI Research", "Industry News", "Policy"]
    selected_topic = st.selectbox("Topic", topics, index=0)
    
    # Date Filter
    st.subheader("Date Range")
    start_date = st.date_input("Start Date", value=None)
    end_date = st.date_input("End Date", value=None)
    
    # Image Settings
    st.subheader("🖼️ Image Settings")
    clip_threshold = st.slider(
        "CLIP Relevance Threshold", 
        min_value=0.0, 
        max_value=0.5, 
        value=0.20, 
        step=0.05,
        help="Higher = stricter filtering. Only images above this score will be shown."
    )
    st.session_state.clip_threshold = clip_threshold
    
    # Construct filters dict
    st.session_state.retrieval_filters = {}
    if selected_topic != "All":
        st.session_state.retrieval_filters["topic"] = selected_topic
    if start_date:
        st.session_state.retrieval_filters["start_date"] = start_date.strftime("%Y-%m-%d")
    if end_date:
        st.session_state.retrieval_filters["end_date"] = end_date.strftime("%Y-%m-%d")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message:
            for img in message["images"]:
                st.image(img, width=400)
        if "sources" in message:
            with st.expander("View Sources"):
                for src in message["sources"]:
                    st.markdown(f"**[{src['title']}]({src['source']})**")
                    if src.get('image_url'):
                         st.image(src['image_url'], width=200)
                    st.text(src['content'][:200] + "...")

if prompt := st.chat_input("Ask about the latest AI news..."):
    if not api_key:
        st.error("Please enter an API Key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get cached retriever
                    retriever = get_retriever()
                    
                    # Check if database has documents - warn user if empty
                    if len(retriever.bm25_docs) == 0:
                        st.warning("⚠️ **No documents in database.** The system will use web search only. "
                                   "For best results, click '🔄 Ingest Data' in the sidebar first.")
                    
                    filters = st.session_state.get("retrieval_filters", {})
                    result = query_rag(prompt, api_key, retriever, filters=filters)
                    answer = result["answer"]
                    sources = result["source_documents"]
                    
                    st.markdown(answer)
                    
                    # Get CLIP-filtered relevant images using the REWRITTEN query (not the original prompt with typos)
                    clip_query = result.get("final_query", prompt)  # Fall back to prompt if no rewrite
                    clip_threshold = st.session_state.get("clip_threshold", 0.20)
                    relevant_images = retriever.get_relevant_images(clip_query, threshold=clip_threshold, max_images=3)
                    
                    if relevant_images:
                        st.subheader("Related Images")
                        cols = st.columns(len(relevant_images))
                        for i, img in enumerate(relevant_images):
                            with cols[i]:
                                st.image(img["image_url"], use_container_width=True)
                                st.caption(f"Score: {img['score']:.2f}")
                    
                    # Prepare source metadata for display
                    source_data = []
                    for doc in sources:
                        meta = doc.metadata
                        src_info = {
                            "title": meta.get("title", "Unknown"),
                            "source": meta.get("source", "#"),
                            "image_url": meta.get("image_url"),
                            "content": doc.page_content
                        }
                        source_data.append(src_info)
                    
                    with st.expander("Verified Sources"):
                        for src in source_data:
                            st.markdown(f"**[{src['title']}]({src['source']})**")
                            st.caption(src['content'][:300] + "...")
                            
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": source_data,
                        "images": [img["image_url"] for img in relevant_images]
                    })
                    
                except Exception as e:
                    st.error(f"Error: {e}")
