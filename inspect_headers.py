import requests
from bs4 import BeautifulSoup

url = "https://www.deeplearning.ai/the-batch/issue-282/"
response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(response.content, 'html.parser')

print("Searching for headers...")
for level in ['h1', 'h2', 'h3']:
    headers = soup.find_all(level)
    print(f"Found {len(headers)} {level} tags.")
    if headers:
        first = headers[0]
        print(f"First {level}: {first.get_text(strip=True)[:50]}")
        print(f"Parent of first {level}: {first.parent.name} class={first.parent.get('class')}")
        print(f"Grandparent of first {level}: {first.parent.parent.name} class={first.parent.parent.get('class')}")
