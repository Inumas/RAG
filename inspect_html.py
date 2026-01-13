import requests
from bs4 import BeautifulSoup

url = "https://www.deeplearning.ai/the-batch/issue-282/"
response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(response.content, 'html.parser')

article = soup.find('article') or soup.find('div', class_='prose')

if article:
    print(f"Found container: {article.name} class={article.get('class')}")
    print("Direct children tags:")
    for child in article.children:
        if child.name:
            print(f"- {child.name}")
            if child.name in ['div', 'section']:
                print(f"  Nested children in {child.name}: {[c.name for c in child.children if c.name]}")
else:
    print("No article container found.")
