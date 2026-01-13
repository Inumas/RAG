import requests
from bs4 import BeautifulSoup

url = "https://www.deeplearning.ai/the-batch/issue-282/"
response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(response.content, 'html.parser')

print("All h1 tags:")
for i, h1 in enumerate(soup.find_all('h1')):
    print(f"{i+1}. {h1.get_text(strip=True)} (Parent: {h1.parent.name})")

print("\nAll h2 tags:")
for i, h2 in enumerate(soup.find_all('h2')):
    print(f"{i+1}. {h2.get_text(strip=True)} (Parent: {h2.parent.name})")

print("\nAll h4 tags:")
for i, h4 in enumerate(soup.find_all('h4')):
    print(f"{i+1}. {h4.get_text(strip=True)} (Parent: {h4.parent.name})")
