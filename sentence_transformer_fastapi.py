import PyPDF2
import nltk.data
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import openai
from Bio import Entrez
from PyPDF2 import PdfReader
import io 
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

Entrez.email = ' '
Entrez.api_key = ' '
openai.api_key = ' '
app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 


"""
Input: question: str, context: str
Output: Answer: str
Functionality: It takes the question and context and give the precise answer using GPT-3.5 Turbo model
"""
def tenant_and_landlord_from_chatgpt(question, context):
    prompt=f'The question is: {question}. The context is: {context}.'
    try:
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a legal assistant that give the answer in maximum 7 words given the question context pair"},
                {"role": "user", "content":prompt }
            ],
        temperature = 0.2
        )
        precise_answer = response['choices'][0]['message']['content']

    except Exception as e:
        print(f'Error : {str(e)}')
    if precise_answer is not None:
        return precise_answer
    else:
        return None



"""
Input: question: str, answer: str
Output: Precise Answer: str
Functionality: It takes the question and answer and give the precise answer using GPT-3.5 Turbo model
"""
def preciseanswer_from_chatgpt(question, answer):
    prompt=f'The question is: {question}. The answer of the question is: {answer}.'
    try:
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful assistant that precise the answer in maximum 7 words given the question answer pair"},
                # {"role": "system", "content": "You are a legal expert that precise the answer to bare minimum given the question answer pair"},
                {"role": "user", "content":prompt }
            ],
        temperature = 0.2
        )
        precise_answer = response['choices'][0]['message']['content']

    except Exception as e:
        print(f'Error : {str(e)}')
    if precise_answer is not None:
        return precise_answer
    else:
        return None


def text_extractor_from_pdf(filepath):
    reader = PdfReader(io.BytesIO(filepath))
    number_of_pages = len(reader.pages)
    text_page = ""
    for i in range(0, number_of_pages):
        page = reader.pages[i]
        text = page.extract_text()
        text_page = text_page +" " + text
    # text_page = text_page.replace("\n"," ")
    # Split the text into a list of words
    words = text_page.split()

    # Get the important words
    important_words = " ".join(words[:300])
    return text_page, important_words

# Split document into sentences using Punkt tokenizer
def split_sentences(context):    
    sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_detector.tokenize(context.strip())
    return sentences

def compute_embeddings(model, context):
    embeddings = model.encode(context, convert_to_tensor=True)
    return embeddings


def get_precise_answer(model, question, document_embedding, sentences):
    question_embedding= model.encode(question, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(question_embedding, document_embedding)
    index = torch.topk(similarity, k=5).indices
    answer=''
    for i in index[0]:
        answer=str(answer)+ "." + sentences[i]
    precise_answer = preciseanswer_from_chatgpt(question, answer)
    return precise_answer

@app.post("/uploadpdf_question_answers/")
async def create_upload_clause_clfr(file: UploadFile = File(...)):

    try:
        contents = await file.read() 
        context, important_words = text_extractor_from_pdf(contents)
        sentences = split_sentences(context)
        # Load the model
        model = torch.load("sentence_transformer.pkl")
        embedding_document = compute_embeddings(model, sentences)
        question_landlord = "Who OWNS the property of this agreement?"
        question_tenant = "What is the name of the Tenant"
        questions=['what is the type of agreement?', 
                'What is the rent amount of the property?',
                'What is the deposit amount?',
                    'what are the measures for deposit protection?',  
                    'what is the type of tenancy?',
                    'What is the term duration of the agreement?',
                    'Is subletting allowed?',
                    'Does the tenant pay the utilities?',
                    ' Is smoking allowed at the property place?',
                    'Who will be liable to pay for repair?',
                    'Are alterations allowed?',
                        'what happens if payment of rent is late?',
                        'what will be the notice period?',
                    "What is the address of the landlord to which the notice should be sent?",  
                    "What is the address of Tenant written in agreement?",
                    "what is the property address for this agreement?",
                    "When will be the start date of tenancy?",
                    "When will be the end date of tenancy?",
                        " Find the name of country whose laws govern this agreement?",
                        "What is the name of the agent?",
                        "When rent payment is due for each month?",
                        "Who is the guarantor of this agreement?",                   
                        "What is the date of execution of this aggrement?"
                        ]
        question_suffix=['Type of agreement',
                        'Rent Amount',
                        'Deposit Amount',
                        'Deposit Protection',
                        'Type of Tenancy',
                        'Term duration',
                        'Is subletting allowed?',
                        'Pay the utilities',
                        'Is smoking allowed?',
                        'Who will be liable to pay for repair?',
                        'Are alterations allowed?',
                        'What happens if payment of rent is late?',
                        'Notice period',
                        'Landlord address',
                        'Tenant address',
                        'Property address',
                        'Tenancy start date',
                        'Tenancy end date',
                        'Govering Law',
                        'Agent',
                        'Rent due date',
                        'Guarantor',
                        'Execution Date'                     
                        ]

        answers={}
        csv_answers={}
        count=0

        for question, code in zip(questions,question_suffix):
            if count<13:
                answer = get_precise_answer(model, question, embedding_document, sentences)
                answers[code]=answer
                csv_answers[code]=answer
                count+=1
            else:
                answer = get_precise_answer(model, question, embedding_document, sentences)
                csv_answers[code]=answer
                count+=1
        answer_landlord =tenant_and_landlord_from_chatgpt(question_landlord, important_words)
        answer_tenant =tenant_and_landlord_from_chatgpt(question_tenant, important_words)
        answers["Lanlord Name"] = answer_landlord
        answers["Tenant Name"] = answer_tenant
        csv_answers["Lanlord Name"] = answer_landlord
        csv_answers["Tenant Name"] = answer_tenant
  

    except Exception as e:
        print(f'Error : {str(e)}')    
    

    return {"answers":answers,"Landlord":answer_landlord, "Tenant":answer_tenant, "csv_answers": csv_answers}
    
