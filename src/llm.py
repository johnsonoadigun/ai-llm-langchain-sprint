from langchain_ollama import ChatOllama

def get_llm(model: str = "llama3.2:3b", temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=model, temperature=temperature, base_url="http://localhost:11434")