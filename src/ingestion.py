import requests
from bs4 import BeautifulSoup
import os
import re
from datetime import datetime
import dateparser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from unstructured.partition.html import partition_html
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://www.deeplearning.ai/the-batch/"
SITEMAP_URLS = [
    'https://www.deeplearning.ai/sitemap-0.xml',
    'https://www.deeplearning.ai/sitemap-1.xml',
]
EXCLUDE_PATTERNS = ['/tag/', '/about/', '/subscribe', '/page/']

def get_article_links():
    """Scrapes the main page for article links."""
    try:
        response = requests.get(BASE_URL, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        links = set()
        # Finding all links that contain 'the-batch' but excluding tags, generic pages
        for a in soup.find_all('a', href=True):
            href = a['href']
            if '/the-batch/' in href and not any(x in href for x in ['/tag/', '/about/', '/subscribe']):
                if href.startswith('/'):
                    href = "https://www.deeplearning.ai" + href
                links.add(href)
        
        # Filter out the main page itself if present
        if BASE_URL in links:
            links.remove(BASE_URL)
        if "https://www.deeplearning.ai/the-batch" in links:
             links.remove("https://www.deeplearning.ai/the-batch")
             
        # Limit to 5-10 articles for the demo to save time/resources
        return list(links)[:10] 
    except Exception as e:
        print(f"Error scraping main page: {e}")
        return []


def get_all_batch_urls_from_sitemap():
    """
    Discover ALL Batch URLs from sitemap, sorted by lastmod date (most recent first).
    Returns list of (url, lastmod_date) tuples.
    """
    all_urls = []
    
    for sitemap_url in SITEMAP_URLS:
        try:
            print(f"Fetching {sitemap_url}...")
            response = requests.get(sitemap_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'xml')
            
            for url_tag in soup.find_all('url'):
                loc = url_tag.find('loc')
                lastmod = url_tag.find('lastmod')
                
                if loc:
                    url = loc.text
                    # Filter for Batch content only
                    if '/the-batch/' in url and not any(ex in url for ex in EXCLUDE_PATTERNS):
                        # Skip main page
                        if url.rstrip('/') != 'https://www.deeplearning.ai/the-batch':
                            lastmod_date = lastmod.text if lastmod else "1970-01-01"
                            all_urls.append((url, lastmod_date))
        except Exception as e:
            print(f"Error fetching {sitemap_url}: {e}")
    
    # Sort by lastmod date (most recent first)
    all_urls.sort(key=lambda x: x[1], reverse=True)
    
    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for url, date in all_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append((url, date))
    
    print(f"Found {len(unique_urls)} unique Batch URLs")
    return unique_urls


def get_existing_urls_from_db():
    """Get all URLs already in the database."""
    try:
        from database import get_vectorstore
        vs = get_vectorstore()
        results = vs.get()
        
        existing_urls = set()
        for metadata in results.get('metadatas', []):
            if metadata and 'source' in metadata:
                existing_urls.add(metadata['source'])
        
        return existing_urls
    except Exception as e:
        print(f"Error getting existing URLs: {e}")
        return set()


def classify_content_type(url, text_content=""):
    """Classify URL into content type based on URL pattern and page content."""
    # Check URL pattern first
    if '/issue-' in url:
        # Extract issue number
        match = re.search(r'/issue-(\d+)', url)
        issue_num = int(match.group(1)) if match else None
        return "issue", issue_num
    
    return "article", None


def extract_topic(text_preview):
    """
    Uses an LLM to categorize the article topic.
    Cost-effective: Uses GPT-4o-mini on just the first 1000 chars.
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        template = """
        Analyze the following article excerpt and assign a SINGLE HIGH-LEVEL TOPIC from this list:
        
        [Generative AI, LLMs, Computer Vision, Robotics, AI Ethics, Hardware, AI Research, Industry News, Policy]
        
        If none fit perfectly, choose the closest one. Return ONLY the topic name.
        
        Excerpt:
        {excerpt}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm
        
        # Limit input to save tokens
        excerpt = text_preview[:1000]
        response = chain.invoke({"excerpt": excerpt})
        
        topic = response.content.strip()
        # Basic validation (if LLM chats too much, fallback)
        if len(topic) > 30: 
            return "General AI"
        return topic
        
    except Exception as e:
        print(f"Error extracting topic: {e}")
        return "Unknown"

def scrape_article(url):
    """
    Scrapes a single article using Unstructured.io for robust layout parsing.
    Extracts metadata including Topic (via LLM) and standardized Date.
    """
    try:
        # 1. Fetch content
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        response.raise_for_status()
        
        # 2. Extract Metadata via BeautifulSoup (lighter than Unstructured for meta tags)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Title
        h1 = soup.find('h1')
        title = h1.get_text(strip=True) if h1 else soup.title.get_text(strip=True) if soup.title else "No Title"
        
        # Image
        image_url = ""
        img_tag = soup.find('img', src=lambda x: x and 'charonhub.deeplearning.ai/content/images' in x)
        if img_tag:
            image_url = img_tag['src']
            
        # Date Standardization
        raw_date = None
        date_meta = soup.find('meta', property='article:published_time')
        if date_meta:
            raw_date = date_meta.get('content')
        else:
            time_elem = soup.find('time', {'datetime': True})
            if time_elem:
                raw_date = time_elem.get('datetime')
        
        # ISO 8601 Formatting & Timestamp
        publish_date = ""
        timestamp = 0
        if raw_date:
            parsed_date = dateparser.parse(raw_date)
            if parsed_date:
                publish_date = parsed_date.strftime('%Y-%m-%d') # ISO 8601
                timestamp = int(parsed_date.timestamp())
        
        # Content Type & Issue
        content_type, issue_number = classify_content_type(url)

        # 3. Parse Content with Unstructured
        # Use partitioning to get clean text elements, handling layout better than raw soup
        elements = partition_html(text=response.text)
        
        # Reconstruct text from elements
        # Filter for meaningful text (NarrativeText, Title, ListItem)
        # Unstructured elements have .text attribute
        full_text = "\n\n".join([str(el) for el in elements if hasattr(el, 'text') and len(str(el)) > 20])
        
        if not full_text.strip():
             print(f"No meaningful text found for {url}")
             return []

        # 4. Extract Topic (using the first chunk of text)
        topic = extract_topic(full_text)
        
        metadata = {
            "source": url,
            "title": title,
            "image_url": image_url,
            "publish_date": publish_date,
            "timestamp": timestamp, # Numeric field for filtering
            "content_type": content_type,
            "topic": topic
        }
        if issue_number:
            metadata["issue_number"] = issue_number
            
        doc = Document(page_content=full_text, metadata=metadata)
        
        # Auto-index image with CLIP if image_url exists
        if image_url:
            try:
                _index_image_with_clip(url, image_url, metadata)
            except Exception as e:
                print(f"  Warning: CLIP indexing failed for image: {e}")
        
        return [doc]

    except Exception as e:
        print(f"Error scraping article {url}: {e}")
        return None


def _index_image_with_clip(source_url: str, image_url: str, metadata: dict):
    """
    Index a single image with CLIP embedding.
    Called automatically during article ingestion.
    """
    import hashlib
    
    try:
        from clip_embeddings import get_clip_embedder
        from database import index_image_embeddings
        
        embedder = get_clip_embedder()
        embedding = embedder.embed_image(image_url)
        
        if embedding:
            image_data = [{
                'id': hashlib.md5(image_url.encode()).hexdigest(),
                'embedding': embedding,
                'metadata': {
                    'image_url': image_url,
                    'source': source_url,
                    'title': metadata.get('title', ''),
                    'topic': metadata.get('topic', ''),
                    'issue_number': metadata.get('issue_number'),
                }
            }]
            index_image_embeddings(image_data)
            print(f"  ✓ CLIP indexed: {image_url[:50]}...")
    except ImportError:
        # CLIP not installed, skip silently
        pass
    except Exception as e:
        print(f"  CLIP indexing error: {e}")

def load_data(mode="issues", start_issue=None, end_issue=None, max_articles=None, skip_existing=True):
    """
    Main function to load data with multiple modes.
    
    Args:
        mode: "sitemap" (all from sitemap), "issues" (issue range), "recent" (main page)
        start_issue: Start issue number (for mode="issues")
        end_issue: End issue number (for mode="issues")
        max_articles: Maximum articles to ingest (for mode="sitemap")
        skip_existing: Skip URLs already in database (for mode="sitemap")
    
    Returns:
        List of Document objects
    """
    documents = []
    
    if mode == "sitemap":
        # Get all URLs from sitemap (sorted by date, most recent first)
        all_urls = get_all_batch_urls_from_sitemap()
        
        # Filter out existing URLs if requested
        if skip_existing:
            existing_urls = get_existing_urls_from_db()
            print(f"Found {len(existing_urls)} URLs already in database")
            all_urls = [(url, date) for url, date in all_urls if url not in existing_urls]
            print(f"After filtering: {len(all_urls)} new URLs to ingest")
        
        # Limit to max_articles if specified
        if max_articles:
            all_urls = all_urls[:max_articles]
        
        print(f"Ingesting {len(all_urls)} articles...")
        
        for i, (url, lastmod) in enumerate(all_urls):
            print(f"[{i+1}/{len(all_urls)}] Scraping {url}...")
            article_docs = scrape_article(url)
            if article_docs:
                documents.extend(article_docs)
            else:
                print(f"  Skipping (no content or error)")
            
            # Progress indicator every 50 articles
            if (i + 1) % 50 == 0:
                print(f"Progress: {i+1}/{len(all_urls)} articles processed, {len(documents)} documents created")
    
    elif mode == "issues" and start_issue is not None and end_issue is not None:
        print(f"Scraping issues {start_issue} to {end_issue}...")
        for i in range(start_issue, end_issue + 1):
            url = f"{BASE_URL}issue-{i}/"
            print(f"Scraping {url}...")
            article_docs = scrape_article(url)
            if article_docs:
                documents.extend(article_docs)
            else:
                print(f"Skipping issue {i} (404 or empty)")
    
    else:
        # Default behavior: Scrape main page links
        links = get_article_links()
        print(f"Found {len(links)} articles.")
        for link in links:
            print(f"Scraping {link}...")
            article_docs = scrape_article(link)
            if article_docs:
                documents.extend(article_docs)
                
    return documents


def load_incremental(max_new=100):
    """
    Load only NEW articles that aren't in the database yet.
    
    Args:
        max_new: Maximum number of new articles to ingest
        
    Returns:
        List of Document objects (only new ones)
    """
    return load_data(mode="sitemap", max_articles=max_new, skip_existing=True)

if __name__ == "__main__":
    docs = load_data()
    print(f"Successfully scraped {len(docs)} documents.")
