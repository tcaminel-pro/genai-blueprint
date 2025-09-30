from genai_tk.core.embeddings_factory import EmbeddingsFactory, get_embeddings

print(EmbeddingsFactory.known_items())
embedder = get_embeddings()

# or select by tag from a configuration YAML file:
# azure_embedder = get_embeddings(embeddings_tag="azure")

# Generate embedding for first sentence
vector_1 = embedder.embed_documents(["Hello"])
