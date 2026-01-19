"""Database statistics script for evaluation."""
import sys
sys.path.append('src')
from database import get_vectorstore

vs = get_vectorstore()
results = vs.get()

print(f"Total documents: {len(results['ids'])}")

# Gather statistics
sources = set()
topics = {}
issue_numbers = set()
content_types = {}
with_timestamp = 0

for m in results['metadatas']:
    if not m:
        continue
    if m.get('source'):
        sources.add(m['source'])
    if m.get('topic'):
        topics[m['topic']] = topics.get(m['topic'], 0) + 1
    if m.get('issue_number'):
        issue_numbers.add(m['issue_number'])
    if m.get('content_type'):
        content_types[m['content_type']] = content_types.get(m['content_type'], 0) + 1
    if m.get('timestamp'):
        with_timestamp += 1

print(f"Unique sources (articles): {len(sources)}")
print(f"Documents with timestamp: {with_timestamp}")
print(f"Issue numbers: {sorted(issue_numbers)[-10:]} (showing last 10)")
print(f"Topics: {topics}")
print(f"Content types: {content_types}")
print(f"\nSample sources:")
for s in list(sources)[:5]:
    print(f"  - {s}")
