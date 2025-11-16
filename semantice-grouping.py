"""
semantic_grouping.py
---------------------
Groups semantically similar queries using OpenAI embeddings
and simple K-Means clustering.
"""

from openai import OpenAI
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import os

# -------------------------------
# 1️⃣ Setup
# -------------------------------
# Make sure you have your OpenAI API key set as:
# export OPENAI_API_KEY="sk-..."
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Example queries
queries = [
    "king",
    "queen",
    "man",
    "woman",
    "car"
]

# -------------------------------
# 2️⃣ Generate Embeddings
# -------------------------------
print("Generating embeddings...")
response = client.embeddings.create(model="text-embedding-3-small", input=queries)
embeddings = np.array([d.embedding for d in response.data])

# -------------------------------
# 3️⃣ Cluster Similar Queries
# -------------------------------
# Choose number of clusters (you can tune this)
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42,)
labels = kmeans.fit_predict(embeddings)
 
# -------------------------------
# 4️⃣ Identify Representative Query (Cluster Centroid)
# -------------------------------
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
representative_queries = [queries[idx] for idx in closest]


# -------------------------------
# 5️⃣ Label Clusters (Auto-summary using LLM)
# -------------------------------
def summarize_cluster(queries_in_cluster):
    joined = "\n".join(queries_in_cluster)
    prompt = (
        f"Summarize the common topic of these user queries in a short label, "
        f"using the format [Category][Subtopic][Keyword][Intent]:\n{joined}"
        "Category must be one of [Product/ no_product]. "
    )
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return completion.choices[0].message.content.strip()


cluster_labels = []
for i in range(n_clusters):
    cluster_queries = [q for q, label in zip(queries, labels) if label == i]
    label_text = summarize_cluster(cluster_queries)
    cluster_labels.append((label_text, cluster_queries))

# -------------------------------
# 6️⃣ Display Results
# -------------------------------
print("\n=== Semantic Groups ===")
for label, group in cluster_labels:
    print(f"\n{label}")
    for q in group:
        print(f"  - {q}")

# Optional: show representative query
print("\nRepresentative queries per cluster:")
for i, rep in enumerate(representative_queries):
    print(f"  Cluster {i + 1}: {rep}")
