import requests
from bs4 import BeautifulSoup
import os
from langchain_core.documents import Document

BASE_URL = "https://www.deeplearning.ai/the-batch/"

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

def scrape_article(url):
    """Scrapes a single article for title, text, and main image."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Title
        h1 = soup.find('h1')
        title = h1.get_text(strip=True) if h1 else soup.title.get_text(strip=True) if soup.title else "No Title"
        article = soup.find('article') or soup.find('div', class_='prose')
        
        if not article:
             print(f"No content found for {url}")
             return []

        # Image
        image_url = None
        img_tag = soup.find('img', src=lambda x: x and 'charonhub.deeplearning.ai/content/images' in x)
        if img_tag:
            image_url = img_tag['src']

        # Parse sections
        content_chunks = []
        current_section = "Introduction"
        current_text = []

        # Strategy: Find the container that holds the headers.
        # 1. Look for headers inside the article
        headers = article.find_all(['h1', 'h2', 'h3', 'h4'])
        
        if headers:
            # Check if all headers are in the same container? No, just use find_all order.
            pass

        # Strategy: Flatten the document structure by finding all relevant block tags in order.
        # This handles arbitrary nesting/containers (divs, sections, etc).
        # We explicitly exclude containers like div/article/section to avoid duplication 
        # (getting container text + child text).
        # We use 'li' for lists and 'p' for text.
        
        elements = article.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'blockquote', 'figure'])
        
        for element in elements:
            # Skip if no text and not a figure (sometimes figures have no text but are important landmarks? 
            # Actually figures are usually images, handled separately or ignored for text)
            
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                # Save previous section if it has text
                if current_text:
                    full_text = " ".join(current_text).strip()
                    if full_text:
                        content_chunks.append(Document(
                            page_content=full_text,
                            metadata={
                                "source": url,
                                "title": title,
                                "image_url": image_url or "",
                                "section_header": current_section
                            }
                        ))
                # Start new section
                current_section = element.get_text(separator=' ', strip=True)
                current_text = []
            elif element.name in ['p', 'li', 'blockquote']:
                # For blockquote, we might duplicate if it contains p. 
                # Check if it has p children?
                if element.name == 'blockquote' and element.find('p'):
                    continue # Let the p child handle it
                
                text = element.get_text(separator=' ', strip=True)
                if text:
                    current_text.append(text)
        
        # Add the last section
        if current_text:
            full_text = " ".join(current_text).strip()
            if full_text:
                content_chunks.append(Document(
                    page_content=full_text,
                    metadata={
                        "source": url,
                        "title": title,
                        "image_url": image_url or "",
                        "section_header": current_section
                    }
                ))

        return content_chunks
    except Exception as e:
        print(f"Error scraping article {url}: {e}")
        return None

def load_data(start_issue=None, end_issue=None):
    """
    Main function to load data.
    If start_issue and end_issue are provided, scrapes that range.
    Otherwise scrapes the main page.
    """
    documents = []
    
    if start_issue is not None and end_issue is not None:
        print(f"Scraping issues {start_issue} to {end_issue}...")
        for i in range(start_issue, end_issue + 1):
            url = f"{BASE_URL}issue-{i}/"
            print(f"Scraping {url}...")
            article_docs = scrape_article(url)
            if article_docs:
                # Add metadata to indicate it's an issue
                for doc in article_docs:
                    doc.metadata["type"] = "issue"
                    doc.metadata["issue_number"] = i
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

if __name__ == "__main__":
    docs = load_data()
    print(f"Successfully scraped {len(docs)} documents.")
