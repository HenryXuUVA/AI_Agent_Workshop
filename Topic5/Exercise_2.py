# Exercise 2
import os
from google.colab import userdata
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

#
try:
    api_key = userdata.get('OPENAI_API_KEY')
    os.environ["OPENAI_API_KEY"] = api_key
    print("✅ \n")
except userdata.SecretNotFoundError:
    print("❌ ")

#
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

def ask_model_direct(question: str) -> str:
    prompt = f"Answer this question:\n{question}\n\nAnswer:"
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

#
QUERIES_MODEL_T = [
    "How do I adjust the carburetor on a Model T?",
    "What is the correct spark plug gap for a Model T Ford?",
    "How do I fix a slipping transmission band?",
    "What oil should I use in a Model T engine?",
]

QUERIES_CR = [
    "What did Mr. Flood have to say about Mayor David Black in Congress on January 13, 2026?",
    "What mistake did Elise Stefanik make in Congress on January 23, 2026?",
    "What is the purpose of the Main Street Parity Act?",
    "Who in Congress has spoken for and against funding of pregnancy centers?",
]

print("="*60)
print("Model T Ford Queries")
print("="*60)
for q in QUERIES_MODEL_T:
    print(f"\nQuestion: {q}")
    print("-" * 40)
    print(ask_model_direct(q))

print("\n" + "="*60)
print("Congressional Record Queries")
print("="*60)
for q in QUERIES_CR:
    print(f"\nQuestion: {q}")
    print("-" * 40)
    print(ask_model_direct(q))