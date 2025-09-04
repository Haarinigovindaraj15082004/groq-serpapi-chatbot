```
# 🤖 Groq + SerpAPI Chatbot

This project is a simple chatbot that uses:
- **Groq LLMs** (via `langchain_groq`) for natural language processing  
- **SerpAPI** for web search results  
- **LangGraph** for chaining together the search and answer generation  

The chatbot takes a user’s question, searches the web for relevant information, and then summarizes the results into a clear answer using Groq.

---

## 🚀 Features
- Uses **SerpAPI** to fetch real-time Google search results  
- Uses **Groq LLMs** (`llama-3.1-8b-instant`) to generate concise answers  
- Built with **LangChain + LangGraph** for modular and scalable workflow  
- Interactive **command-line chatbot** interface  

---

## 📂 Project Structure
```

task2/
│-- main.py          # Main chatbot code
│-- .env             # API keys (not pushed to GitHub)
│-- requirements.txt # Python dependencies
│-- README.md        # Project documentation

````

---

## 🔑 Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/groq-serpapi-chatbot.git
cd groq-serpapi-chatbot
````

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add API keys

Create a `.env` file in the project root:

```ini
GROQ_API_KEY=your_groq_api_key_here
SERPAPI_API_KEY=your_serpapi_api_key_here
```

### 5️⃣ Run the chatbot

```bash
python main.py
```

---

## 💻 Usage Example

```
🤖 Groq + SerpAPI chatbot ready! Type 'exit' to quit.

You: Where is SRM University located?
Chatbot: SRM University is located in Kattankulathur, near Chennai, Tamil Nadu, India.
```

---

## 📦 requirements.txt

```
langchain
langchain-community
langchain-groq
langgraph
python-dotenv
google-search-results
```

---

## ⚠️ Notes

* Make sure your **GROQ\_API\_KEY** and **SERPAPI\_API\_KEY** are valid.
* Do **NOT** push `.env` to GitHub (keep your keys private).
* If you see model errors, update the Groq model in `main.py` to a supported one.

---
