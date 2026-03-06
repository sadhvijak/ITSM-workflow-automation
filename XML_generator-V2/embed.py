

import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# Load the Nomic-Embed-Text-v1 model from Hugging Face
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# Initialize persistent ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection(name="xml_pattern_embeddings")

# Path to the JSON file
json_file_path = "enhanced_xml_patterns_with recordtrigger_for_rag2.json"

# Load JSON content
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Function to generate embedding using Nomic open-source model
def generate_embedding(text):
    return model.encode(text).tolist()

if "rag_chunks" in data:
    rag_chunks = data["rag_chunks"]
    
    for idx, chunk in enumerate(rag_chunks):
        chunk_id = f"chunk_{idx}_{chunk.get('type', 'unknown')}"
        chunk_text = chunk.get("description", json.dumps(chunk))

        # Generate embedding vector
        embedding = generate_embedding(chunk_text)

        # Store in ChromaDB
        collection.add(
            ids=[chunk_id],
            documents=[chunk_text],
            embeddings=[embedding],
            metadatas=[{
                "type": chunk.get("type", "unknown"),
                "element": chunk.get("element", ""),
                "source": "generic_xml_patterns_for_rag.json"
            }]
        )
        
        print(f"✅ Embedded chunk {idx + 1}/{len(rag_chunks)}: {chunk.get('type')}")

full_content = json.dumps(data["full_patterns"], ensure_ascii=False)
full_embedding = generate_embedding(full_content)

collection.add(
    ids=["full_patterns"],
    documents=[full_content],
    embeddings=[full_embedding],
    metadatas=[{"type": "full_patterns", "source": "generic_xml_patterns_for_rag.json"}]
)

print(f"\n🎉 Successfully embedded {len(rag_chunks)} chunks + full patterns into ChromaDB!")
