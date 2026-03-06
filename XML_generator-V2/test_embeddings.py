import chromadb

client = chromadb.PersistentClient(path="./chroma")
collections = client.list_collections()
print("Collections found:", collections)

# If xml_pattern_embeddings exists, check its count
collection = client.get_collection("xml_pattern_embeddings")
print("Total embeddings stored:", collection.count())