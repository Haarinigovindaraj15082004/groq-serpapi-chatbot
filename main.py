import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

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
# 4. Conversation memory
# ---------------------------
conversation_history = []

# ---------------------------
# 5. Search node (silent fail)
# ---------------------------
def search_node(state: ChatState):
    """Call SerpAPI, extract top links, and scrape content from websites silently."""
    query = state["question"]
    search_results = search_tool.results(query)
    urls = [r['link'] for r in search_results['organic_results'][:5]]

    page_texts = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            page_texts.append(docs[0].page_content[:2000])
        except Exception:
            # silently ignore failed URLs
            continue

    combined_content = "\n\n".join(page_texts) if page_texts else "No content found from top URLs."
    return {"search_result": combined_content}

# ---------------------------
# 6. LLM node with chat memory
# ---------------------------
def llm_node(state: ChatState):
    """Generate a polite, friendly response using conversation memory."""
    query = state["question"]
    context = state["search_result"]

    # Build conversation history text
    history_text = ""
    for msg in conversation_history:
        role = "You" if msg["role"] == "user" else "Chatbot"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""
You are a very polite, friendly assistant. 
Respond naturally and conversationally using the conversation history and website content.

Guidelines:
- Always provide clear, concise, and structured answers.
- Avoid irrelevant information or unnecessary details.
- Focus only on what is directly useful to answer the user's question.
- If information is missing, respond politely without apologizing excessively or mentioning failed sources.
- Continue the conversation as if you remember the previous context.
- End responses politely and encourage further questions.
- Do NOT include greetings, personal remarks, or repetitive phrases like "I'm happy to help you" or "you asked earlier".


Conversation History:
{history_text}

Current Question: {query}

Website Content:
{context[:4000]}

    Chatbot:"""

    response = llm.invoke(prompt)
    return {"answer": response.content.strip()}


# ---------------------------
# 7. Build LangGraph
# ---------------------------
graph = StateGraph(ChatState)
graph.add_node("search", search_node)
graph.add_node("llm", llm_node)
graph.set_entry_point("search")
graph.add_edge("search", "llm")
graph.add_edge("llm", END)
app = graph.compile()

# ---------------------------
# 8. Run the Chatbot
# ---------------------------
print("🤖 Groq + SerpAPI Chatbot ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye 👋")
        break

    # Add user message to history
    conversation_history.append({"role": "user", "content": user_input})

    # Run search node
    search_result = search_node({"question": user_input})

    # Run LLM node with memory
    llm_result = llm_node({"question": user_input, "search_result": search_result["search_result"]})

    # Add bot reply to history
    conversation_history.append({"role": "assistant", "content": llm_result["answer"]})

    # Print reply
    print(f"Chatbot: {llm_result['answer']}\n")
