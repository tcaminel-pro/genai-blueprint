
client<llm> DeepseekChat {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key "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"
    model "deepseek/deepseek-chat-v3.1"
    headers {
    }
  }
}


client<llm> Gpt41mini {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key "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"
    model "gpt-4.1-mini"
    headers {
    }
  }
}