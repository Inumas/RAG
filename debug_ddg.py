from langchain_community.tools import DuckDuckGoSearchRun
import traceback

print("Attempting to instantiate DuckDuckGoSearchRun directly...")
try:
    search = DuckDuckGoSearchRun()
    print("Tool instantiated.")
    print("Running search...")
    res = search.invoke("test")
    print(f"Search Result: {res[:50]}")
except Exception:
    traceback.print_exc()
