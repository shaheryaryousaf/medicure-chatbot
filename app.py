import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="🤖",
    layout="centered"
)

# --- APPLICATION ---
def main():
    """Main function to run the Streamlit application."""
    st.title("🤖 PDF Chatbot")
    st.write("Ask any question about the content of your PDF file, and I'll do my best to answer!")

    # --- INITIALIZE CHAT HISTORY ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you with the document today?"}
        ]

    # --- DISPLAY CHAT MESSAGES ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- SETUP RETRIEVALQA CHAIN ---
    try:
        # Initialize embeddings and language model
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.3) # Using a more advanced model
        index_name = "pdf-chatbot-index"

        # Initialize Pinecone Vector Store as a retriever
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 4})

        # Define a more detailed prompt template for beginners with formatting instructions
        prompt_template = """
        You are a friendly and knowledgeable guide, designed to help beginners understand the content of the provided document.
        Your primary goal is to answer the user's questions based *only* on the text given to you in the 'Context' section.

        Please follow these instructions carefully:
        1.  **Analyze the Context**: Read the following 'Context' section, which contains excerpts from the document relevant to the user's question.
        2.  **Answer the Question**: Use the information in the context to answer the 'Question' at the end.
        3.  **Be Helpful for Beginners**: Explain your answer in a clear, simple, and encouraging way. If the user asks about a complex concept, try to break it down for them.
        4.  **Synthesize and Infer**: If a direct answer isn't available in one chunk, combine information from different parts of the context to form a complete answer. It's okay to make a logical inference, but you should base it entirely on the text provided.
        5.  **Stay Grounded**: **Do not** use any information outside of the provided context. If the context does not contain the answer, politely state that the document doesn't seem to cover that specific topic. Avoid making up information at all costs.
        6.  **Format Your Answer**: Before finalizing your response, think about the best way to present the information.
            - If the answer contains a sequence of steps, a list of items, or distinct points, use a Markdown numbered or bulleted list.
            - For explanations or descriptions, use well-structured paragraphs.
            - Your goal is to make the answer as clear and easy to read as possible.

        Context:
        ---
        {context}
        ---

        Question: {question}

        Friendly and Helpful Answer:
        """
        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

        # Create the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"Failed to initialize the application. Please check your API keys and configurations. Error: {e}")
        return

    # --- USER INPUT AND RESPONSE ---
    if prompt := st.chat_input("Ask your question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = qa_chain.invoke({"query": prompt})
                    response = result["result"]

                    # Optionally, display source documents
                    # with st.expander("Show Sources"):
                    #     for doc in result["source_documents"]:
                    #         st.write(f"- Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:150]}...")

                except Exception as e:
                    response = f"An error occurred: {e}"

                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    # Check for API keys before running the app
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        st.error("🚨 API keys for OpenAI or Pinecone are not set. Please create a .env file with your keys.")
    else:
        main()
