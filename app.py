from __future__ import annotations

import json
import os
from typing import Any, Dict

import streamlit as st
from groq import Groq
from dotenv import load_dotenv

from src.data_utils import build_retrieval_corpus, load_policy_documents, load_qa_pairs, load_ticket_data
from src.rag_engine import SimpleRAG

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


@st.cache_data
def get_policy_context() -> str:
    docs = load_policy_documents()
    context_blocks: list[str] = []
    for name, text in docs.items():
        clean_text = " ".join(text.split())
        context_blocks.append(f"## {name}\n{clean_text[:2500]}")
    return "\n\n".join(context_blocks)


@st.cache_resource
def get_rag_store() -> SimpleRAG:
    tickets = load_ticket_data()
    qa_pairs = load_qa_pairs()
    policy_docs = load_policy_documents()
    corpus = build_retrieval_corpus(tickets, qa_pairs, policy_docs)
    return SimpleRAG(corpus)


def get_groq_key() -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        api_key = st.session_state.get("groq_api_key_input", "").strip()

    if api_key:
        st.session_state["groq_client"] = Groq(api_key=api_key)
    return api_key


def call_groq(prompt: str) -> str:
    client = st.session_state.get("groq_client")
    if client is None:
        raise RuntimeError("Groq client is not configured. Please enter your Groq API key.")

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a banking operations expert."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


def parse_json_response(raw_text: str) -> Dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


def build_analysis_prompt(query: str, amount: float | None, merchant: str, merchant_category: str, transaction_type: str, city: str, hour: int, day: str, is_international: bool, velocity_flag: bool, geo_flag: bool, high_amount_flag: bool, retrieved_docs: list[dict]) -> str:
    policy_context = "\n\n".join(
        f"Source: {doc['source']} | Title: {doc['title']}\n{doc['text']}" for doc in retrieved_docs[:5]
    )

    return f"""
You are a senior banking operations and fraud specialist for an Indian bank.
Use the retrieval results below as the main evidence and answer the customer query.
Return ONLY valid JSON in this format:
{{
  "intent": "Fraud|Loan|KYC|Account Access|General",
  "sentiment": "Calm|Neutral|Concerned|Frustrated|Urgent|Angry|Anxious",
  "risk_level": "Low|Medium|High",
  "action": "Clear next step for the bank team",
  "response": "Customer-facing response in plain English"
}}

Customer query:
{query}

Context details:
- Amount: {amount or 'Not provided'}
- Merchant: {merchant}
- Merchant category: {merchant_category}
- Transaction type: {transaction_type}
- City: {city}
- Hour: {hour}
- Day: {day}
- Is international: {is_international}
- Velocity anomaly: {velocity_flag}
- Geo anomaly: {geo_flag}
- High amount anomaly: {high_amount_flag}

Retrieved knowledge:
{policy_context}
"""


def analyze_query(query: str, amount: float | None, merchant: str, merchant_category: str, transaction_type: str, city: str, hour: int, day: str, is_international: bool, velocity_flag: bool, geo_flag: bool, high_amount_flag: bool) -> Dict[str, Any]:
    api_key = get_groq_key()
    if not api_key:
        raise RuntimeError("A Groq API key is required to run this project. Add it in the sidebar or set GROQ_API_KEY in the environment.")

    rag_store = get_rag_store()
    retrieved_docs = rag_store.retrieve(query, top_k=5)

    prompt = build_analysis_prompt(
        query=query,
        amount=amount,
        merchant=merchant,
        merchant_category=merchant_category,
        transaction_type=transaction_type,
        city=city,
        hour=hour,
        day=day,
        is_international=is_international,
        velocity_flag=velocity_flag,
        geo_flag=geo_flag,
        high_amount_flag=high_amount_flag,
        retrieved_docs=retrieved_docs,
    )

    raw_response = call_groq(prompt)
    result = parse_json_response(raw_response)

    return {
        "intent": result.get("intent", "General"),
        "sentiment": result.get("sentiment", "Neutral"),
        "risk_level": result.get("risk_level", "Low"),
        "action": result.get("action", "Route the case to the relevant support queue."),
        "response": result.get("response", "We are reviewing your issue."),
        "retrieved_docs": retrieved_docs,
        "fraud_probability": 0.0,
        "fraud_label": "N/A",
        "intent_accuracy": "N/A",
        "sentiment_accuracy": "N/A",
        "fraud_accuracy": "N/A",
    }


def main() -> None:
    st.set_page_config(page_title="Banking Support with Groq + RAG", page_icon="🏦", layout="wide")
    st.title("Banking Support & Fraud Intelligence")
    st.caption("Uses Groq LLM + FAISS semantic retrieval from policies and past cases.")

    with st.sidebar:
        st.header("API access")
        st.write("Enter your Groq API key below or set the environment variable GROQ_API_KEY.")
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            key="groq_api_key_input",
            help="Paste your Groq API key. This project is configured to work only with Groq.",
        )
        if api_key:
            st.session_state["groq_client"] = Groq(api_key=api_key)
        st.caption("Status: Ready" if api_key else "Status: Waiting for API key")

        st.header("Customer query")
        query_text = st.text_area(
            "Customer query",
            value="I see a transaction of ₹10,000 I didn't make",
            help="Describe the banking issue, fraud alert, loan problem, or KYC request.",
        )
        amount = st.number_input("Amount (₹)", min_value=0.0, value=10000.0, step=100.0)
        merchant = st.text_input("Merchant name", value="UnknownMerchant")
        merchant_category = st.text_input("Merchant category", value="Unknown")
        transaction_type = st.selectbox("Transaction type", ["UPI", "Debit Card", "Credit Card", "Net Banking", "ATM", "NEFT"])
        city = st.text_input("City", value="Delhi")
        hour = st.slider("Hour of day", 0, 23, 10)
        day = st.selectbox("Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        is_international = st.checkbox("International transaction")
        velocity_flag = st.checkbox("Velocity anomaly")
        geo_flag = st.checkbox("Geo anomaly")
        high_amount_flag = st.checkbox("High amount")

        analyze_button = st.button("Analyze query", use_container_width=True)

    if analyze_button:
        if not api_key:
            st.warning("Add a Groq API key before running the analysis.")
            return

        try:
            with st.spinner("Retrieving relevant policy and case data via RAG and generating the response..."):
                result = analyze_query(
                    query=query_text,
                    amount=amount,
                    merchant=merchant,
                    merchant_category=merchant_category,
                    transaction_type=transaction_type,
                    city=city,
                    hour=hour,
                    day=day,
                    is_international=is_international,
                    velocity_flag=velocity_flag,
                    geo_flag=geo_flag,
                    high_amount_flag=high_amount_flag,
                )

            st.subheader("Live result")
            col1, col2, col3 = st.columns(3)
            col1.metric("Intent", result["intent"])
            col2.metric("Sentiment", result["sentiment"])
            col3.metric("Risk", result["risk_level"])

            st.write("### Suggested response")
            st.info(result["response"])

            st.write("### Suggested action")
            st.success(result["action"])

            st.write("### Retrieved evidence")
            for index, doc in enumerate(result["retrieved_docs"][:5], start=1):
                with st.expander(f"{index}. {doc['source'].upper()} — {doc['title']}"):
                    st.write(doc["text"])

        except Exception as exc:  # pragma: no cover - UI error handling
            st.error(f"Groq request failed: {exc}")
            st.write("Check that the API key is valid and that your internet connection is available.")
    else:
        st.info("Enter a customer query and click Analyze query to generate a response from Groq using semantic retrieval.")


if __name__ == "__main__":
    main()
