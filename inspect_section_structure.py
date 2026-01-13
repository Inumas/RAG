import requests
from bs4 import BeautifulSoup

url = "https://www.deeplearning.ai/the-batch/issue-282/"
response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(response.content, 'html.parser')

# Find the h1 "2025 Beckons"
target_h1 = soup.find('h1', string=lambda t: t and "2025 Beckons" in t)

if target_h1:
    parent = target_h1.parent
    print(f"Parent tag: {parent.name} class={parent.get('class')}")
    print("Siblings in parent:")
    for child in parent.children:
        if child.name:
            print(f"- {child.name} (text preview: {child.get_text(separator=' ', strip=True)[:30]}...)")
else:
    print("Target Header not found.")
