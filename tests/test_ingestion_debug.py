import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ingestion import scrape_article

# Test URL (a recent batch issue or article)
url = "https://www.deeplearning.ai/the-batch/issue-282/" 

print(f"Testing scraping for: {url}")
docs = scrape_article(url)

if docs:
    print(f"Success! Retrieved {len(docs)} chunks.")
    for i, doc in enumerate(docs):
        print(f"--- Chunk {i+1} ---")
        print(f"Header: {doc.metadata.get('section_header')}")
        print(f"Content Preview: {doc.page_content[:100]}...")
else:
    print("Failed to scrape.")
