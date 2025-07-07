# Phase 3 Upgraded: pubmed_utils.py

from query_faiss import search_mesh_terms
from pubmed_search import search_pubmed

BLACKLIST_TERMS = {
    "burnout", "wellbeing", "quality of life", "financial capacity",
    "social support", "social workers", "occupation", "employment"
}

BLACKLIST_TITLE_KEYWORDS = {
    "burnout", "financial", "social worker", "employment", "wellbeing"
}

RELEVANT_TITLE_KEYWORDS = [
    "human", "throat", "pharyngitis", "streptococcus", "infection", "clinical"
]

def score_article_relevance(title, mesh_term, main_condition=None):
    score = 0
    title = title.lower()
    mesh_term = mesh_term.lower()
    if mesh_term in title:
        score += 2
    if main_condition and main_condition.lower() in title:
        score += 2
    if any(symptom in title for symptom in ["pain", "infection", "treatment", "syndrome", "disease", "disorder"]):
        score += 1
    if any(keyword in title for keyword in RELEVANT_TITLE_KEYWORDS):
        score += 2
    # Loosen: allow articles with score >= 1
    return score

def get_evidence_links(condition_name, symptom_text=None, top_k=4, debug=False):
    """
    Returns up to top_k PubMed links most relevant to the predicted condition (human-focused).
    Falls back to user symptom text if not enough links are found.
    """
    def _search_and_score(search_term, main_condition):
        mesh_matches = search_mesh_terms(search_term)
        filtered_terms = [
            term for term, _ in mesh_matches
            if term.lower() not in BLACKLIST_TERMS
        ][:3]
        seen_links = set()
        scored_articles = []
        fallback_articles = []
        for term in filtered_terms:
            links = search_pubmed(term)
            for link in links:
                link_lower = link.lower()
                if any(bad in link_lower for bad in BLACKLIST_TITLE_KEYWORDS):
                    continue
                if link in seen_links:
                    continue
                seen_links.add(link)
                score = score_article_relevance(link, term, main_condition)
                if score >= 1:  # Loosened from >0 to >=1
                    scored_articles.append((score, link))
                else:
                    fallback_articles.append(link)
        scored_articles.sort(reverse=True)
        top_links = [article for _, article in scored_articles[:top_k]]
        if not top_links and fallback_articles:
            top_links = fallback_articles[:top_k]
        return top_links

    # 1. Try with predicted condition name
    links = _search_and_score(condition_name, condition_name)
    if debug:
        print(f"Links for condition '{condition_name}': {links}")
    # 2. Fallback: Try with symptom_text if not enough links
    if (not links or len(links) < top_k) and symptom_text and symptom_text != condition_name:
        more_links = _search_and_score(symptom_text, condition_name)
        # Avoid duplicates
        links = links + [l for l in more_links if l not in links]
        if debug:
            print(f"Links for symptoms '{symptom_text}': {more_links}")
    return links[:top_k]
