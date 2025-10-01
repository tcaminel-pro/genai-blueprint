from genai_tk.core.embeddings_factory import EmbeddingsFactory
from genai_tk.utils.config_mngr import global_config

model = global_config().get("llm.models")
print(model)

print(EmbeddingsFactory.known_list())
