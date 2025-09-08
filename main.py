import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.document_loaders import WebBaseLoader, PlaywrightURLLoader
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
import re

def extract_website(query: str):
    # Simple regex to detect URL in query
    match = re.search(r'(https?://[^\s]+)', query)
    return match.group(0) if match else None

# ---------------------------
# Conversation memory (short-term)
# ---------------------------
conversation_history = []
MAX_HISTORY = 6  # only last 6 messages to avoid confusion
turn_counter = 0  # track numbered turns
current_entity = {"type": None, "name": None}  # entity tracker
first_question = None  # store first user question
cached_results = {}  # cache search results per entity

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
    query = state["question"].strip()
    website = extract_website(query)

    # Handle vague follow-ups
    vague_queries = ["tell me more", "more details", "i want to know more",
                     "tell more about it", "more about the movie",
                     "more about the company", "details please"]
    if any(vq in query.lower() for vq in vague_queries) and current_entity["name"]:
        query = current_entity["name"]
        website = None  # treat vague query as general entity

    # ---------------------------
    # If query has a website → scrape that directly
    # ---------------------------
    if website:
        urls = [website]

    # ---------------------------
    # If no website → search authoritative sources
    # ---------------------------
    else:
        # Add trusted sites if query is about a movie or company
        if "movie" in query.lower():
            query += " site:wikipedia.org OR site:imdb.com"
        elif "company" in query.lower():
            query += " site:linkedin.com OR site:crunchbase.com OR site:glassdoor.com"

        # Use SerpAPI to get top 10 results
        search_results = search_tool.results(query)
        urls = [r['link'] for r in search_results.get("organic_results", [])[:10]]
        # ---------------------------
        # Filter only trusted sources
        trusted_sources = ["wikipedia.org", "imdb.com", "linkedin.com", "crunchbase.com"]
        urls = [url for url in urls if any(t in url for t in trusted_sources)]
        # ---------------------------

    # ---------------------------
    # 3️⃣ Scrape URLs
    # ---------------------------
    page_texts = []
    for url in urls:
        success = False
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            if docs and docs[0].page_content.strip():
                page_texts.append(docs[0].page_content[:2000])
                success = True
        except Exception:
            pass

        # Playwright fallback
        if not success:
            try:
                loader = PlaywrightURLLoader(
                    urls=[url],
                    remove_selectors=["script", "style"],
                    browser_type="chromium",
                    headless=True,
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/117 Safari/537.36",
                    timeout=90000
                )
                docs = loader.load()
                if docs and docs[0].page_content.strip():
                    page_texts.append(docs[0].page_content[:2000])
            except Exception:
                continue

    combined_content = "\n\n".join(page_texts) if page_texts else ""
    if not combined_content:
        return {"search_result": "No reliable content could be retrieved from the web. Please clarify or try again."}

    # Update entity tracking
    if not any(vq in state["question"].lower() for vq in vague_queries):
        current_entity["name"] = state["question"]
        current_entity["type"] = "general"
        cached_results[state["question"]] = combined_content

    return {"search_result": combined_content}

# ---------------------------
# 5. LLM Node
# ---------------------------
def llm_node(state: ChatState):
    """Generate a polite, accurate response using short-term memory."""

    query = state["question"]
    context = state["search_result"]

    # Only include user messages in numbered history to avoid repetition
    numbered_history = [
        f"{i}. {msg['content']}"
        for i, msg in enumerate(conversation_history[-MAX_HISTORY:], start=1)
        if msg["role"] == "user"
    ]

    chat_prompt = [
        {"role": "system", "content": f"""You are a polite, helpful and respectful assistant.
I am giving you the conversation history below in numbered form:
{chr(10).join(numbered_history)}

Strict Conversation rules:
1. Always remember the user’s earlier questions and answers in this chat.
2. Only connect vague follow-ups (like "tell me more", "details", "about it")
   to the most recent meaningful question or entity.
3. If the new question looks specific (like a new company, movie, or person name),
   treat it as a completely new entity and DO NOT connect it to earlier topics.
4. If the user asks a vague follow-up like "tell me more", "more details",
   "about the movie", "about the company", etc., connect it back to the most
   recent meaningful question or entity they asked about.
   Example:
     - Q1: "name of thalapathy 69"
     - Q2: (irrelevant question)
     - Q3: "tell more about the movie"
       → You must understand Q3 refers to the movie from Q1 ("Thalapathy 69" / "Jana Nayagan")
5. When such follow-ups occur, fetch fresh web information about that entity
   instead of treating it as a brand-new query.
6. Do not mix unrelated topics. Stick to either the new question or the last tracked entity.
7. If you are uncertain about what the user means, politely ask them to clarify
   instead of guessing or hallucinating.
8. Use ONLY the retrieved web context when answering.
9. Do not make up facts or speculate. 
10. Give direct and concise answer.
11. If web results are limited, combine available snippets with prior conversation context.
12. Give results in professionaly manner (DO NOT use # or * for bold elements)

You are a web search assistant. You are given multiple sources of information. 
- If the sources agree, provide a concise answer. 
- If the sources conflict, mention the conflict clearly. 
Example: "Some sources say Nelson is the director, while others mention H. Vinoth."
- Prefer details from authoritative sites like Wikipedia, IMDb, or official announcements.

"""}
    ]

    chat_prompt.append({"role": "user", "content": query})

    if context and "No reliable content" not in context:
        chat_prompt.append({
            "role": "system",
            "content": f"Extra web context:\n{context[:10000]}"  # increase slice for more context
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

    # Increment turn counter
    turn_counter += 1

    # Save numbered user input
    conversation_history.append({"role": "user", "content": f"[{turn_counter}] {user_input}"})

    # Run search node
    search_result = search_node({"question": user_input})

    # Run LLM node
    llm_result = llm_node({
        "question": user_input,
        "search_result": search_result["search_result"]
    })

    # Save numbered assistant reply
    conversation_history.append({"role": "assistant", "content": f"[{turn_counter}] {llm_result['answer']}"})

    # Special handling for first question inquiry
    if "first question" in user_input.lower():
        print(f"Chatbot: You first asked: '{first_question}'\n")
        continue

    # Print reply
    print(f"Chatbot: {llm_result['answer']}\n")

