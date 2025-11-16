
import re
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity



# ---------------------- Normalization ----------------------
def normalize_text(text: str) -> str:
    """
    Normalize text:
    - Lowercase English letters
    - Keep Japanese characters unchanged
    """
    def lower_en(match):
        return match.group(0).lower()
    
    text = re.sub(r'[A-Za-z]+', lower_en, text)
    return text.strip()

# ---------------------- Jina Embedding ----------------------
def get_embedding(text: str):
    normalized = normalize_text(text)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }

    payload = {
        "model": "jina-embeddings-v3",    # multilingual + strong Japanese support
        "input": normalized
    }

    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers=headers,
        json=payload
    )

    data = response.json()
    return np.array(data["data"][0]["embedding"])

# ---------------------- Similarity ----------------------
def similarity_score(text1: str, text2: str) -> float:
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    return cosine_similarity([emb1], [emb2])[0][0]

# ---------------------- Example ----------------------
if __name__ == "__main__":
    pairs = [
        ("IQOM", "アイコム"),
        ("Tell me about IQOM", "IQOMについて教えてください。"),
        ("Iqom", "アイコム"),
        ("IQOMとは何ですか？", "What is ICOM?"),
        ("IQOM", "iqom"),
        ("tell me about iqom", "What is ICOM?"),
        ("tell me about iqom", "detail IQOM"),
    ]

    print("🔍 English-lowercase (Japanese untouched) + Jina similarity:\n")
    for t1, t2 in pairs:
        score = similarity_score(t1, t2)
        n1, n2 = normalize_text(t1), normalize_text(t2)
        print(f"'{t1}' → '{t2}'  | similarity={score:.4f}")
