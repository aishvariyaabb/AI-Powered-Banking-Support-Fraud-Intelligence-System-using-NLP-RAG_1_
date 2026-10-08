# AI-Powered Banking Support & Fraud Intelligence System

This project combines:
- NLP-based intent and sentiment classification
- A lightweight RAG-style retriever over support tickets, QA pairs, and policy documents
- A fraud-risk classifier using transaction data
- A Streamlit-based support dashboard

## Files
- `app.py`: main Streamlit application
- `src/data_utils.py`: loaders and preprocessing helpers
- `src/ml_models.py`: intent, sentiment, and fraud models
- `src/rag_engine.py`: retrieval and response generation logic
- `requirements.txt`: Python dependencies

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Demo flow
1. Enter a customer issue such as "I see a transaction of ₹10,000 I didn't make"
2. Optionally provide transaction metadata
3. The app returns:
   - predicted intent
   - detected sentiment
   - retrieved policy/FAQ evidence
   - suggested response
   - risk level and fraud probability
   - recommended support action

## Environment variables

- **GROQ_API_KEY**: Required. Your Groq API key for the LLM backend. The app reads this from the environment or from the Streamlit sidebar.
- **GROQ_MODEL**: Optional. Override the model used by the Groq client (defaults in code if not set).

Create a local `.env` file at the project root or set the variables in your shell. An example file is provided at `.env.example`.

Windows PowerShell example:

```powershell
$env:GROQ_API_KEY = "your_groq_api_key_here"
streamlit run files\app.py
```

Or using the included `.env` file (the app calls `dotenv.load_dotenv()` automatically):

```bash
cp files/.env.example files/.env
# edit files/.env and add your key
streamlit run files/app.py
```
