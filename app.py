import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from vector import setup_vectorstore, get_retriever
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load API Key
load_dotenv()

# Setup model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Natural prompt for friendly answers
template = """
You are a professional car reviewer who has read and analyzed thousands of customer reviews.

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

# Streamlit UI setup
st.set_page_config(page_title="Car Review Assistant", page_icon="🚗")
st.title("🚗 Car Review Q&A Assistant")
st.markdown("Ask a question based on thousands of real car reviews!")

# Load vectorstore and retriever
with st.spinner("Searching for reviews"):
    vectorstore = setup_vectorstore()
    retriever = get_retriever(vectorstore, k=10)

# User input
question = st.text_input("Ask your question (e.g. 'Is the 2018 Honda Civic reliable?')")

# Button to trigger response
if st.button("Get Answer") and question:
    with st.spinner("Retrieving relevant reviews and generating answer..."):
        docs = retriever.get_relevant_documents(question)
        review_texts = "\n\n".join([doc.page_content for doc in docs])
        result = chain.invoke({"reviews": review_texts, "question": question})

    # Show final result
    st.success("✅ Here's what our expert found:")
    st.markdown(result.content)

    # Optional debug/review output
    with st.expander("🔍 Show relevant review snippets"):
        for doc in docs:
            st.markdown(f"- {doc.page_content[:300]}...")
