# import os
# import sys
# from flask import Flask, request, jsonify, session, render_template
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from langchain_community.vectorstores import Chroma
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# # Set environment variable to avoid issues with duplicate libraries
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'
# app.config['SESSION_TYPE'] = 'filesystem'

# def load_model_and_tokenizer():
#     try:
#         checkpoint = "MBZUAI/LaMini-T5-738M"
#         tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#         model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
#         return tokenizer, model
#     except Exception as e:
#         print(f"Error in load_model_and_tokenizer: {e}", file=sys.stderr)
#         raise

# def llm_pipeline():
#     try:
#         tokenizer, model = load_model_and_tokenizer()
#         pipe = pipeline(
#             'text2text-generation',
#             model=model,
#             tokenizer=tokenizer,
#             max_length=256,
#             do_sample=True,
#             temperature=0.3,
#             top_p=0.95
#         )
#         return HuggingFacePipeline(pipeline=pipe)
#     except Exception as e:
#         print(f"Error in llm_pipeline: {e}", file=sys.stderr)
#         raise

# def qa_llm():
#     try:
#         llm = llm_pipeline()
#         embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         db = Chroma(persist_directory="db", embedding_function=embeddings)
#         retriever = db.as_retriever()
#         qa = RetrievalQA.from_chain_type(
#             llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
#         )
#         return qa
#     except Exception as e:
#         print(f"Error in qa_llm: {e}", file=sys.stderr)
#         raise

# def process_answer(instruction):
#     try:
#         qa = qa_llm()
#         generated_text = qa(instruction)
#         answer = generated_text['result']
#         return answer, generated_text
#     except Exception as e:
#         print(f"Error in process_answer: {e}", file=sys.stderr)
#         raise

# @app.route('/')
# def index():
#     if 'history' not in session:
#         session['history'] = []
#     return render_template('index.html', history=session['history'])

# @app.route('/ask', methods=['POST'])
# def ask():
#     data = request.get_json()
#     question = data.get('question', '')
#     if question:
#         try:
#             answer, metadata = process_answer(question)
#             session['history'].append({"message": question, "is_user": True})
#             session['history'].append({"message": answer, "is_user": False})
#             return jsonify({"answer": answer, "history": session['history']})
#         except Exception as e:
#             return jsonify({"error": f"Error in processing: {e}"}), 500
#     else:
#         return jsonify({"error": "No question provided"}), 400

# @app.route('/history', methods=['GET'])
# def get_history():
#     return jsonify({"history": session.get('history', [])})

# if __name__ == '__main__':
#     app.run(debug=True



# import sys
# import os
# from flask import Flask, render_template, request, jsonify
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from langchain_community.vectorstores import Chroma
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# # Set environment variable to avoid issues with duplicate libraries
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# app = Flask(__name__)

# tokenizer = None
# model = None

# def load_model_and_tokenizer():
#     global tokenizer, model
#     try:
#         if tokenizer is None or model is None:
#             checkpoint = "MBZUAI/LaMini-T5-738M"
#             tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#             model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
#     except Exception as e:
#         print(f"Error in load_model_and_tokenizer: {e}", file=sys.stderr)
#         raise

# def llm_pipeline():
#     try:
#         load_model_and_tokenizer()
#         if tokenizer is None or model is None:
#             raise ValueError("Tokenizer or model is None after loading.")
        
#         pipe = pipeline(
#             'text2text-generation',
#             model=model,
#             tokenizer=tokenizer,
#             max_length=256,
#             do_sample=True,
#             temperature=0.3,
#             top_p=0.9
#         )
#         return HuggingFacePipeline(pipeline=pipe)
#     except Exception as e:
#         print(f"Error in llm_pipeline: {e}", file=sys.stderr)
#         raise

# def qa_llm():
#     try:
#         llm = llm_pipeline()
#         if llm is None:
#             raise ValueError("Failed to create LLM pipeline.")
        
#         embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         db = Chroma(persist_directory="db", embedding_function=embeddings)
#         retriever = db.as_retriever()
#         qa = RetrievalQA.from_chain_type(
#             llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
#         )
#         return qa
#     except Exception as e:
#         print(f"Error in qa_llm: {e}", file=sys.stderr)
#         raise

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/ask', methods=['POST'])
# def ask():
#     try:
#         question = request.form['question']
#         answer, metadata = process_answer(question)
#         return jsonify({'answer': answer})
#     except Exception as e:
#         return jsonify({'error': str(e)})

# def process_answer(instruction):
#     try:
#         qa = qa_llm()
#         if qa is None:
#             raise ValueError("Failed to create QA instance.")
        
#         generated_text = qa(instruction)
#         answer = generated_text['result']
#         return answer, generated_text
#     except Exception as e:
#         print(f"Error in process_answer: {e}", file=sys.stderr)
#         raise

# if __name__ == '__main__':
#     app.run(debug=True)




# import os
# import sys
# from flask import Flask, request, jsonify, session, render_template
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from langchain_community.vectorstores import Chroma
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# # Set environment variable to avoid issues with duplicate libraries
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'
# app.config['SESSION_TYPE'] = 'filesystem'

# def load_model_and_tokenizer():
#     try:
#         checkpoint = "MBZUAI/LaMini-T5-738M"
#         tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#         model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
#         return tokenizer, model
#     except Exception as e:
#         print(f"Error in load_model_and_tokenizer: {e}", file=sys.stderr)
#         raise

# def llm_pipeline():
#     try:
#         tokenizer, model = load_model_and_tokenizer()
#         pipe = pipeline(
#             'text2text-generation',
#             model=model,
#             tokenizer=tokenizer,
#             max_length=256,
#             do_sample=True,
#             temperature=0.3,
#             top_p=0.95
#         )
#         return HuggingFacePipeline(pipeline=pipe)
#     except Exception as e:
#         print(f"Error in llm_pipeline: {e}", file=sys.stderr)
#         raise

# def qa_llm():
#     try:
#         llm = llm_pipeline()
#         embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         db = Chroma(persist_directory="db", embedding_function=embeddings)
#         retriever = db.as_retriever()
#         qa = RetrievalQA.from_chain_type(
#             llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
#         )
#         return qa
#     except Exception as e:
#         print(f"Error in qa_llm: {e}", file=sys.stderr)
#         raise

# def process_answer(instruction):
#     try:
#         qa = qa_llm()
#         generated_text = qa(instruction)
#         answer = generated_text['result']
#         return answer, generated_text
#     except Exception as e:
#         print(f"Error in process_answer: {e}", file=sys.stderr)
#         raise

# @app.route('/')
# def index():
#     if 'history' not in session:
#         session['history'] = []
#     return render_template('index.html', history=session['history'])

# @app.route('/ask', methods=['POST'])
# def ask():
#     data = request.get_json()
#     question = data.get('question', '')
#     if question:
#         try:
#             answer, metadata = process_answer(question)
#             session['history'].append({"message": question, "is_user": True})
#             session['history'].append({"message": answer, "is_user": False})
#             return jsonify({"answer": answer, "history": session['history']})
#         except Exception as e:
#             return jsonify({"error": f"Error in processing: {e}"}), 500
#     else:
#         return jsonify({"error": "No question provided"}), 400

# @app.route('/history', methods=['GET'])
# def get_history():
#     return jsonify({"history": session.get('history', [])})

# if __name__ == '__main__':
#     app.run(debug=True)




import sys
import os
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# Set environment variable to avoid issues with duplicate libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = Flask(__name__)

tokenizer = None
model = None

def load_model_and_tokenizer():
    global tokenizer, model
    try:
        if tokenizer is None or model is None:
            checkpoint = "MBZUAI/LaMini-T5-738M"
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    except Exception as e:
        print(f"Error in load_model_and_tokenizer: {e}", file=sys.stderr)
        raise

def llm_pipeline():
    try:
        load_model_and_tokenizer()
        if tokenizer is None or model is None:
            raise ValueError("Tokenizer or model is None after loading.")
        
        pipe = pipeline(
            'text2text-generation',
            model=model,
            tokenizer=tokenizer,
            max_length=256,
            do_sample=True,
            temperature=0.3,
            top_p=0.9
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        print(f"Error in llm_pipeline: {e}", file=sys.stderr)
        raise

def qa_llm():
    try:
        llm = llm_pipeline()
        if llm is None:
            raise ValueError("Failed to create LLM pipeline.")
        
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        question = request.form['question']
        answer, metadata = process_answer(question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)})

def process_answer(instruction):
    try:
        qa = qa_llm()
        if qa is None:
            raise ValueError("Failed to create QA instance.")
        
        generated_text = qa(instruction)
        answer = generated_text['result']
        return answer, generated_text
    except Exception as e:
        print(f"Error in process_answer: {e}", file=sys.stderr)
        raise

if __name__ == '__main__':
port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)




