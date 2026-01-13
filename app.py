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
    
    ingestion_mode = st.radio("Ingestion Mode", ["Recent Articles", "Historical Issues"])
    
    start_issue = 330
    end_issue = 334
    
    if ingestion_mode == "Historical Issues":
        c1, c2 = st.columns(2)
        start_issue = c1.number_input("Start Issue", min_value=1, value=330)
        end_issue = c2.number_input("End Issue", min_value=1, value=334)
    
    if st.button("🔄 Ingest Data"):
        if not api_key:
            st.error("Please enter an OpenAI API Key first.")
        else:
            with st.status("Ingesting data...", expanded=True) as status:
                # Only clear if we are doing a fresh start? Or maybe append?
                # For this demo, we'll keep it simple: option to clear
                if st.checkbox("Clear existing database?", value=False):
                    st.write("Clearing old database...")
                    clear_database()
                
                if ingestion_mode == "Historical Issues":
                    st.write(f"Scraping Issues {start_issue} to {end_issue}...")
                    docs = load_data(start_issue=start_issue, end_issue=end_issue)
                else:
                    st.write("Scraping recent articles...")
                    docs = load_data()
                
                st.write(f"Scraped {len(docs)} documents.")
                
                if docs:
                    st.write("Indexing into Vector DB...")
                    index_documents(docs)
                else:
                    st.warning("No documents found.")
                
                status.update(label="Ingestion Complete!", state="complete", expanded=False)
            st.success("Data ready!")

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
                    result = query_rag(prompt, api_key, retriever)
                    answer = result["answer"]
                    sources = result["source_documents"]
                    
                    st.markdown(answer)
                    
                    # Prepare source metadata for display
                    source_data = []
                    images_to_show = []
                    for doc in sources:
                        meta = doc.metadata
                        src_info = {
                            "title": meta.get("title", "Unknown"),
                            "source": meta.get("source", "#"),
                            "image_url": meta.get("image_url"),
                            "content": doc.page_content
                        }
                        source_data.append(src_info)
                        if meta.get("image_url"):
                            images_to_show.append(meta.get("image_url"))
                    
                    # Deduplicate images
                    images_to_show = list(set(images_to_show))
                    
                    if images_to_show:
                        st.subheader("Related Images")
                        cols = st.columns(len(images_to_show))
                        for i, img in enumerate(images_to_show):
                            with cols[i]:
                                st.image(img, use_container_width=True)
                    
                    with st.expander("Verified Sources"):
                        for src in source_data:
                            st.markdown(f"**[{src['title']}]({src['source']})**")
                            st.caption(src['content'][:300] + "...")
                            
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": source_data,
                        "images": images_to_show
                    })
                    
                except Exception as e:
                    st.error(f"Error: {e}")
