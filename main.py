import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_community.utilities import SerpAPIWrapper
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ---------------------------
# 1. Load API keys from .env
# ---------------------------
load_dotenv()

# Now env variables are available
groq_key = os.getenv("GROQ_API_KEY")
serpapi_key = os.getenv("SERPAPI_API_KEY")

if not groq_key or not serpapi_key:
    raise ValueError("❌ Missing GROQ_API_KEY or SERPAPI_API_KEY in .env")

# ---------------------------
# 2. Define State
# ---------------------------
class ChatState(TypedDict):
    question: str
    search_result: str
    answer: str

# ---------------------------
# 3. Initialize Tools
# ---------------------------
search_tool = SerpAPIWrapper()   # Will use SERPAPI_API_KEY automatically
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ---------------------------
# 4. Define Graph Nodes
# ---------------------------
def search_node(state: ChatState):
    """Call SerpAPI to fetch web results."""
    query = state["question"]
    results = search_tool.run(query)
    return {"search_result": results}

def llm_node(state: ChatState):
    """Summarize results using Groq LLM."""
    query = state["question"]
    context = state["search_result"]

    prompt = f"""
    You are a helpful assistant.
    User Question: {query}
    Search Results: {context}
    Summarize and answer clearly.
    If not found, say "I don't know."
    """
    response = llm.invoke(prompt)   # Call Groq LLM with prompt
    return {"answer": response.content}    # Save LLM response in state

# ---------------------------
# 5. Build LangGraph
# ---------------------------
graph = StateGraph(ChatState)

#ADD NODES
graph.add_node("search", search_node)  # Node 1: Search
graph.add_node("llm", llm_node)   # Node 2: Summarize with LLM

# Define execution flow
graph.set_entry_point("search")  # Start at "search" node
graph.add_edge("search", "llm")  # After search → go to LLM
graph.add_edge("llm", END)   # After LLM → end

# Compile graph into an executable "app"
app = graph.compile()

# ---------------------------
# 6. Run the Chatbot
# ---------------------------
print("🤖 Groq + SerpAPI chatbot ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye 👋")
        break

    result = app.invoke({"question": user_input})
    print(f"Chatbot: {result['answer']}\n")
