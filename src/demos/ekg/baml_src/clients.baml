
client<llm> DeepseekChat {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "deepseek/deepseek-chat-v3.1"
    headers {
    }
  }
}


client<llm> Gpt41mini {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "gpt-4.1-mini"
    headers {
    }
  }
}