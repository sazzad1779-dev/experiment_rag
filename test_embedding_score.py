import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory cache structure
cache = {}

# ---------------------- Embedding Helper ----------------------
def get_embedding(text: str) -> list:
    """Generate embedding vector using OpenAI API"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# ---------------------- Cache Operations ----------------------
def add_to_cache(query: str, response_text: str):
    """Add a query, its embedding, and response to cache"""
    emb = get_embedding(query)
    cache[query] = {
        "embedding": emb,
        "response": response_text
    }


def find_similar_query(new_query: str):
    """Compare new query against all cache items and return all scores"""
    if not cache:
        return []

    new_emb = get_embedding(new_query)
    results = []

    for query, data in cache.items():
        score = cosine_similarity([new_emb], [data["embedding"]])[0][0]
        results.append((query, score, data["response"]))

    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def get_response(query: str, threshold: float = 0.85):
    """Main function that retrieves or generates response based on threshold"""
    results = find_similar_query(query)

    if not results:
        print("🧠 Cache empty — generating new response.")
        response = f"Generated response for: '{query}'"
        add_to_cache(query, response)
        return response

    print(f"\n🔎 Similarity Trace for: '{query}'")
    for (q, score, _) in results:
        print(f" → {score:.4f} with '{q}'")

    best_match, best_score, best_response = results[0]

    if best_score >= threshold:
        print(f"✅ Cache HIT (similarity={best_score:.2f}) → '{best_match}'")
        return best_response
    else:
        print(f"❌ Cache MISS (max similarity={best_score:.2f} < threshold={threshold})")
        response = f"Generated new response for: '{query}'"
        add_to_cache(query, response)
        return response


# ---------------------- Demo Run ----------------------
if __name__ == "__main__":
    # Seed some cache data
    add_to_cache("tell me about iqom", "Product X is durable with a 4000mAh battery.")
    add_to_cache("describe me about iqom.", "Product Y features a 4K display and 8GB RAM.")
    add_to_cache("give me about iqom", "Product Z lasts up to 12 hours on a full charge.")

    # Try different queries and observe similarity trace
    test_queries = [
        "Give me the specs of product Y",
        "Explain product X battery capacity",
        "How long does product Z last?",
        "Describe product Q"
    ]

    threshold = 0.85  # adjust this to test cache sensitivity

    for q in test_queries:
        print("\n" + "=" * 80)
        print("Query:", q)
        response = get_response(q, threshold)
        print("Response:", response)
