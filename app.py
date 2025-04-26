import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from vector import setup_vectorstore, get_retriever
from langchain_openai import ChatOpenAI
import os
import openai

def validate_openai_key(key: str) -> bool:
    try:
        client = openai.OpenAI(api_key=key)
        _ = client.models.list()
        return True
    except Exception as e:
        return False

# Streamlit UI setup
st.set_page_config(page_title="Car Review Assistant", page_icon="🚗")
st.title("🚗 Car Review Q&A Assistant")
st.markdown("Ask a question based on thousands of real car reviews!")

# Step 1: Ask for API Key
api_key = st.text_input("🔑 Enter your OpenAI API key", type="password")

if api_key:
    if not validate_openai_key(api_key):
        st.error("❌ Invalid or expired OpenAI API key. Please try again.")
        st.stop()
    else:
        st.success("✅ Your OpenAI API key has been validated!")
        os.environ["OPENAI_API_KEY"] = api_key
else:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()


# Step 3: Initialize model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Step 4: Create Prompt
template = """
You are a professional car reviewer who has read and analyzed hundreds of thousands of customer reviews.

You must only answer questions related to cars, car models, reliability, performance, features, comparisons, or any topic directly connected to car reviews.

If the user's question is not about cars or car reviews, respond with:
"I'm designed to answer questions only about cars and car reviews. Please ask me something car-related."

Based on the following user-written reviews, synthesize a clear, helpful, and natural-sounding answer to the car-related question below.

---
Customer Reviews:
{reviews}

---
User Question:
{question}

---
Your Expert Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Load vectorstore and retriever
with st.spinner("Loading car review data..."):
    vectorstore = setup_vectorstore()
    retriever = get_retriever(vectorstore, k=10)

# User input
question = st.text_input("📝 Ask your question (e.g. 'Is the 2018 Honda Civic reliable?')")

if st.button("Get Answer") and question:
    with st.spinner("Retrieving relevant reviews and generating answer..."):
        docs = retriever.get_relevant_documents(question)
        review_texts = "\n\n".join([doc.page_content for doc in docs])
        result = chain.invoke({"reviews": review_texts, "question": question})

    st.success("✅ Here's what our expert found:")
    st.markdown(result.content)

    with st.expander("🔍 Show relevant review snippets"):
        for doc in docs:
            st.markdown(f"- {doc.page_content[:300]}...")
