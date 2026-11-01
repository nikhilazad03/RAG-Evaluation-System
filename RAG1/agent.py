import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from objects import Answer
from indexing import vector_store

load_dotenv()

# Verified active model on Groq
# Available model from your key's model list
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

structured_llm = llm.with_structured_output(Answer)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a question answering assistant for a PDF.
Base your answers only on the retrieved information below.
If the retrieved information does not contain the answer, say that the PDF does not provide enough information to answer the question.

Context:
{context}"""),
    ("human", "{question}"),
])

rag_chain = prompt_template | structured_llm

def ask_question(question: str) -> dict:
    """Answers user question using direct similarity search + structured output."""
    docs = vector_store.similarity_search(question, k=2)
    retrieved_chunks = [doc.page_content for doc in docs]
    context_str = "\n\n".join(retrieved_chunks)

    result = rag_chain.invoke({
        "context": context_str,
        "question": question
    })

    return {
        "answer": result.answer,
        "context": retrieved_chunks
    }


if __name__ == "__main__":
    test_question = "What is the diet of penguins?"
    res = ask_question(test_question)
    print("\n--- Agent Result ---")
    print(f"Answer: {res['answer']}\n")
    print("Context:")
    for i, c in enumerate(res['context'], 1):
        print(f"[{i}] {c}\n")