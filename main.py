import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ---------------------------
# Conversation memory (short-term)
# ---------------------------
conversation_history = []
MAX_HISTORY = 6  # only last 6 messages to avoid confusion
current_entity = {"type": None, "name": None}  # entity tracker
first_question = None  # store first user question

# ---------------------------
# 1. Load API keys
# ---------------------------
load_dotenv()
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
search_tool = SerpAPIWrapper()
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ---------------------------
# 4. Search Node
# ---------------------------
def search_node(state: ChatState):
    """Call SerpAPI, extract top links, and scrape content from websites."""

    query = state["question"].strip().lower()

    # Handle vague follow-ups
    vague_queries = ["tell me more", "more details", "i want to know more",
                     "tell more about it", "more about the movie",
                     "more about the company", "details please"]

    if any(vq in query for vq in vague_queries) and current_entity["name"]:
        query = current_entity["name"]

    # Perform search
    search_results = search_tool.results(query)
    urls = []
    if search_results and "organic_results" in search_results:
        urls = [r['link'] for r in search_results["organic_results"][:5]]

    page_texts = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            page_texts.append(docs[0].page_content[:2000])
        except Exception:
            continue

    combined_content = "\n\n".join(page_texts) if page_texts else "No content found from top URLs."

    # Update entity if query looks specific
    if len(state["question"].split()) > 2 and "No content" not in combined_content:
        current_entity["name"] = state["question"]
        current_entity["type"] = "general"

    return {"search_result": combined_content}

# ---------------------------
# 5. LLM Node
# ---------------------------
def llm_node(state: ChatState):
    """Generate a polite, accurate response using short-term memory."""

    query = state["question"]
    context = state["search_result"]

    # Use last MAX_HISTORY messages
    messages = conversation_history[-MAX_HISTORY:]

    chat_prompt = [
        {"role": "system", "content": """You are a polite, helpful assistant.
Rules:
1. Use recent conversation to stay on topic.
2. If the user asks vague follow-ups (like "tell me more"), use last entity.
3. Never guess; if unsure, ask the user to clarify.
4. Keep answers accurate and grounded in search results."""}
    ]

    for msg in messages:
        chat_prompt.append({"role": msg["role"], "content": msg["content"]})

    chat_prompt.append({"role": "user", "content": query})

    if context and context != "No content found from top URLs.":
        chat_prompt.append({
            "role": "system",
            "content": f"Extra web context:\n{context[:4000]}"
        })

    response = llm.invoke(chat_prompt)
    return {"answer": response.content.strip()}

# ---------------------------
# 6. Build LangGraph
# ---------------------------
graph = StateGraph(ChatState)
graph.add_node("search", search_node)
graph.add_node("llm", llm_node)
graph.set_entry_point("search")
graph.add_edge("search", "llm")
graph.add_edge("llm", END)
app = graph.compile()

# ---------------------------
# 7. Run Chatbot
# ---------------------------
print("🤖 Groq + SerpAPI Chatbot ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye 👋")
        break

    # Store first question
    if first_question is None:
        first_question = user_input

    # Save user input to memory
    conversation_history.append({"role": "user", "content": user_input})

    # Run search node
    search_result = search_node({"question": user_input})

    # Run LLM node
    llm_result = llm_node({
        "question": user_input,
        "search_result": search_result["search_result"]
    })

    # Save assistant reply
    conversation_history.append({"role": "assistant", "content": llm_result["answer"]})

    # Special handling for first question inquiry
    if "first question" in user_input.lower():
        print(f"Chatbot: You first asked: '{first_question}'\n")
        continue

    # Print reply
    print(f"Chatbot: {llm_result['answer']}\n")
