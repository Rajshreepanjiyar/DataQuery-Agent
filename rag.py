"""
rag.py

This module is responsible for building and saving the vector index for schema retrieval.
It gets database schema metadata from schema.py, constructs schema text documents per table
enriched with domain descriptions and keywords, generates embeddings using SentenceTransformer,
builds a FAISS index, and saves both the FAISS index ('schema.index') and schema documents
metadata ('schema_documents.pkl').

Workflow:
1. Retrieve database schema metadata from schema.py.
2. Build descriptive textual representations for each table including table descriptions, columns, and domain keywords.
3. Encode documents using SentenceTransformer ('all-MiniLM-L6-v2').
4. Normalize vectors and index using FAISS (IndexFlatIP for Cosine Similarity).
5. Persist 'schema.index' and 'schema_documents.pkl' to disk.
"""

import pickle
import faiss
from sentence_transformers import SentenceTransformer
from schema import get_schema

# Domain descriptions to enrich semantic table embeddings
TABLE_DESCRIPTIONS = {
    "actor": "Actors, film performers, movie stars, first and last name of actors.",
    "address": "Physical addresses, street location, district, postal code, city linkage, customer location.",
    "category": "Film categories, genres, action, comedy, drama, sci-fi, horror, classification.",
    "city": "Cities, urban locations, city names, linked to country.",
    "country": "Countries, nations, global locations, India, Canada, USA, geographic region.",
    "customer": "Customers, clients, users, buyers, active members who rent films.",
    "film": "Films, movies, motion pictures, title, description, release year, duration, rating.",
    "film_actor": "Cast mapping linking actors to the films they starred in.",
    "film_category": "Genre mapping linking films to their categories.",
    "film_text": "Film titles and text descriptions for searching.",
    "inventory": "Store inventory stock, physical film copies available at store locations.",
    "language": "Spoken languages for films, audio tracks.",
    "payment": "Payment transactions, customer invoices, revenue, rental charges.",
    "rental": "Film rentals, transactions, checkout dates, return dates, customer rental activity.",
    "staff": "Store staff members, employees, managers.",
    "store": "Physical store locations, inventory managers, branch stores."
}

# Get latest database schema from schema.py
schema = get_schema()

# Convert schema into enriched table documents
documents = []
for table_name, columns in schema.items():
    desc = TABLE_DESCRIPTIONS.get(table_name, "")
    document_text = f"Table: {table_name}\nDescription: {desc}\n\nColumns:\n"
    for column in columns:
        column_name = column[0]
        data_type = column[1]
        document_text += f"{column_name} - {data_type}\n"

    documents.append({
        "table_name": table_name,
        "content": document_text
    })

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
doc_texts = [doc["content"] for doc in documents]
embeddings = model.encode(
    doc_texts,
    convert_to_numpy=True
).astype("float32")

# Normalize embeddings for Cosine Similarity search
faiss.normalize_L2(embeddings)

# Create FAISS index using IndexFlatIP (Cosine Similarity)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "schema.index")

# Save schema documents metadata dictionary/list
with open("schema_documents.pkl", "wb") as file:
    pickle.dump(documents, file)

print("=" * 50)
print("Embeddings and FAISS index created successfully.")
print("Number of table documents:", len(documents))
print("Embedding dimension:", dimension)
print("Vectors stored in index (IndexFlatIP):", index.ntotal)
print("=" * 50)

