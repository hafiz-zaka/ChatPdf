import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
import streamlit as st

# Set environment variable to avoid issues with duplicate libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def load_model_and_tokenizer():
    try:
        checkpoint = "MBZUAI/LaMini-T5-738M"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        return tokenizer, model
    except Exception as e:
        print(f"Error in load_model_and_tokenizer: {e}", file=sys.stderr)
        raise

def llm_pipeline():
    try:
        tokenizer, model = load_model_and_tokenizer()
        pipe = pipeline(
            'text2text-generation',
            model=model,
            tokenizer=tokenizer,
            max_length=256,
            do_sample=True,
            temperature=0.3,
            top_p=0.95
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        print(f"Error in llm_pipeline: {e}", file=sys.stderr)
        raise

def qa_llm():
    try:
        llm = llm_pipeline()
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory="db", embedding_function=embeddings)
        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
        )
        return qa
    except Exception as e:
        print(f"Error in qa_llm: {e}", file=sys.stderr)
        raise

def process_answer(instruction):
    try:
        qa = qa_llm()
        generated_text = qa(instruction)
        answer = generated_text['result']
        return answer, generated_text
    except Exception as e:
        print(f"Error in process_answer: {e}", file=sys.stderr)
        raise

def main():
    st.title("Search Your PDF 🐦📄")
    
    with st.expander("About the App"):
        st.markdown(
            """
            This is a Generative AI powered Question and Answering app that responds to questions about your PDF File.
            """
        )
    
    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_area("Enter your Question", key="input_text")

    if st.button("Ask"):
        if question:
            st.session_state.history.append({"message": question, "is_user": True})
            st.info("Your Question: " + question)
            
            try:
                answer, metadata = process_answer(question)
                st.session_state.history.append({"message": answer, "is_user": False})
                st.write("Answer: " + answer)
               
            except Exception as e:
                st.error(f"Error in processing: {e}")

    if st.session_state.history:
        for chat in st.session_state.history:
            if chat["is_user"]:
                st.markdown(f"**User:** {chat['message']}")
            else:
                st.markdown(f"**AI:** {chat['message']}")

if __name__ == '__main__':
    main()
