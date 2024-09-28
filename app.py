import streamlit as st
import os
from pydantic import BaseModel
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

TAVILY_API_KEY = "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"
GROQ_API_KEY = "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"

class MentalHealthChatbot:
    def __init__(self, tavily_api_key: str, groq_api_key: str, model_name: str):
        self.tavily_api_key = tavily_api_key
        self.groq_api_key = groq_api_key
        self.model_name = model_name

        # Set the API keys in the environment
        os.environ["TAVILY_API_KEY"] = self.tavily_api_key
        os.environ["GROQ_API_KEY"] = self.groq_api_key

        # Initialize the TavilySearchAPIRetriever with a focus on APA content
        self.tavily_retriever = TavilySearchAPIRetriever(
            k=3,
            include_domains=["https://psychcentral.com"],
            exclude_domains=[]
        )

        # Set up the chatbot chain with a mental health focus
        self.prompt = ChatPromptTemplate.from_template(
            """You are a compassionate mental health assistant developed by TextFusion.AI. Your role is to provide supportive and informative responses based on reliable mental health information  Always prioritize the user's well-being and give advice for the same.

            Context : {context}

            User's question or concern: {question}

            Please provide a caring, informative response  If the query involves a serious mental health issue, gently suggest seeking professional help."""
        )

        self.chain = (
            RunnablePassthrough.assign(context=(lambda x: x["question"]) | self.tavily_retriever)
            | self.prompt
            | ChatGroq(temperature=0.2, model_name=self.model_name, groq_api_key=self.groq_api_key)
            | StrOutputParser()
        )

    def respond(self, query: str) -> str:
        try:
            response = self.chain.invoke({"question": query})
            return response
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try rephrasing your question or ask something else."

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# Streamlit app
def main():
    st.set_page_config(
        page_title="Mental Health Assistant",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.title("Mental Health Assistant 🧠💚")
    st.subheader("Powered by TextFusion.AI ")

    # Initialize MentalHealthChatbot
    TAVILY_API_KEY = "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"
    GROQ_API_KEY = "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"

    if not TAVILY_API_KEY or not GROQ_API_KEY:
        st.error("API keys are missing. Please set TAVILY_API_KEY and GROQ_API_KEY environment variables.")
    else:
        chatbot = MentalHealthChatbot(
            tavily_api_key=TAVILY_API_KEY,
            groq_api_key=GROQ_API_KEY,
            model_name="mixtral-8x7b-32768"
        )

        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("How are you feeling today?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chatbot.respond(prompt)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    with st.sidebar:
        st.header("🧠 Mental Health Resources")
        st.markdown("""
        - [APA Help Center](https://www.apa.org/topics/crisis-hotlines)
        - [Find a Psychologist](https://locator.apa.org/)
        - [Mental Health Topics](https://www.apa.org/topics)
        """)
        st.write("Copyright © TextFusion.AI")
        st.write("[Instagram](https://instagram.com/textfusion.ai)")
        st.write("For issues, please contact us on Instagram")

if __name__ == "__main__":
    main()