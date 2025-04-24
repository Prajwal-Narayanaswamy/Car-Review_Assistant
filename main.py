from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import setup_vectorstore, get_retriever  # Import get_retriever
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

template =  """
You are an expert in providing car reviews.

Here are some relevant reviews:
{reviews}

Here is the question to answer:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def main():
    vectorstore = setup_vectorstore()  # Get the vectorstore
    retriever = get_retriever(vectorstore, k=10) # Create retriever *from* vectorstore

    while True:
        print("\n\n-------------------------------------------------------------------------------------------")
        question = input("Ask your question (q to quit): ")
        print("\n\n")
        if question.lower() in ('q', 'quit', 'QUIT', 'Quit'):
            break

        print("[DEBUG] Invoking retriever with user question...")
        reviews = retriever.invoke(question)
        review_texts = "\n\n".join([doc.page_content for doc in reviews])

        print("[DEBUG] Running model inference...")
        result = chain.invoke({"reviews": review_texts, "question": question})
        print("[DEBUG] Model response generated.\n")
        print(result.content)

if __name__ == "__main__":
    main()