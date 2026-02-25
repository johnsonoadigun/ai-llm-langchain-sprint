import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .llm import get_llm

TRIAGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Return ONLY valid JSON. No markdown. No extra text."),
    ("user", """
Extract: priority (low/medium/high), summary, and 3 action_items from the text.

Text:
{text}

Return JSON:
{{"priority":"...", "summary":"...", "action_items":["...","...","..."]}}
""")
])

def triage(text: str, model: str = "llama3.1") -> dict:
    llm = get_llm(model=model, temperature=0.0)
    chain = TRIAGE_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"text": text})
    return json.loads(raw)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.triage \"your text here\"")
        raise SystemExit(1)

    result = triage(sys.argv[1])
    print(json.dumps(result, indent=2))
