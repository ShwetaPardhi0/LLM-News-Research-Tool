# 🤖 LLM-Powered News Research Tool

A research tool where you can input a bunch of article URLs and ask questions. The tool retrieves answers based on those articles using Large Language Models (LLMs), embeddings, and vector search.

---

## 🚀 What It Does

- Accepts **multiple URLs of news articles**
- Loads and extracts article content using **LangChain’s UnstructuredURLLoader**
- Constructs document **embeddings using OpenAI**
- Stores embeddings in **FAISS** for similarity search
- Accepts user questions and returns **LLM-generated answers** based on the most relevant articles

---

## 🧠 Technologies Used

| Tool / Library     | Purpose                                      |
|--------------------|----------------------------------------------|
| **Streamlit**      | Frontend interface                           |
| **LangChain**      | Document loading and LLM integration         |
| **OpenAI API**     | Embeddings + Question Answering (LLMs)       |
| **FAISS**          | Fast Approximate Nearest Neighbor Search     |
| **dotenv**         | Secure handling of API keys                  |
| **Python**         | Programming language used throughout         |

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/ShwetaPardhi0/LLM-News-Research-Tool.git
cd LLM-News-Research-Tool
