# pubmed_search.py
import requests
from bs4 import BeautifulSoup

def search_pubmed(query, max_results=3):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    # Step 1: Get article IDs
    search_url = f"{base_url}esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results
    }
    search_res = requests.get(search_url, params=search_params).json()
    id_list = search_res["esearchresult"]["idlist"]

    if not id_list:
        return []

    # Step 2: Get article details
    fetch_url = f"{base_url}esummary.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "json"
    }
    fetch_res = requests.get(fetch_url, params=fetch_params).json()
    articles = []

    for article_id in id_list:
        try:
            title = fetch_res["result"][article_id]["title"]
            article_link = f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/"
            articles.append(f"[{title}]({article_link})")
        except:
            continue

    return articles
