from django.shortcuts import render, redirect
from .models import Enquiry
from django.http import HttpResponse
from django.core.mail import send_mail
from django.conf import settings

import json
import os
from PyPDF2 import PdfReader
from django.http import JsonResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from django.views.decorators.csrf import csrf_exempt

import requests
from cachetools import TTLCache, cached
from langchain import LLMChain
import logging
import asyncio
import aiohttp
from langchain.chains import LLMChain


# Create your views here.


def main(request):
    if request.method == 'POST':
        name = request.POST['name']
        phone = request.POST['phone']
        email = request.POST['email']
        location = request.POST['location']
        detail = request.POST['detail']

        if name and phone and email and location and detail:
            enquiry = Enquiry(name=name, phone=phone,
                              email=email, location=location, detail=detail)
            enquiry.save()

            # Send confirmation email to user
            send_mail(
                subject='Thank you for your enquiry',
                message=f'Dear {name},\n\nThank you for reaching out to us. We have received your enquiry and will get back to you shortly.\n\nDetails:\nName: {name}\nPhone: {phone}\nEmail: {email}\nLocation: {location}\nDetail: {detail}\n\nBest regards,\nAtharva Innovision',
                from_email=settings.EMAIL_HOST_USER,
                recipient_list=[email],
            )

            # Send notification email to admin
            send_mail(
                subject=f'New Enquiry from {name}',
                message=f'You have received a new enquiry.\n\nDetails:\nName: {name}\nPhone: {phone}\nEmail: {email}\nLocation: {location}\nDetail: {detail}\n\nPlease take the necessary action.',
                from_email=settings.EMAIL_HOST_USER,
                recipient_list=['vgolai196@gmail.com'],
            )

            return redirect('home')  # Redirect to the top of the page
        else:
            return HttpResponse("Please fill in all fields.")
    return render(request, 'main.html')


os.environ["GOOGLE_API_KEY"] = 'AIzaSyC_wIoeWx9WLbOzjVmAhL9g_XP4MxZ--Xc'
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set up cache
cache = TTLCache(maxsize=1000, ttl=3600)

# Global variable for vector store
vector_store = None


def initialize_vector_store():
    global vector_store
    index_file_path = "faiss_index_openai/index.faiss"
    if vector_store is None:
        try:
            if os.path.exists(index_file_path):
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001")
                vector_store = FAISS.load_local(
                    "faiss_index_openai", embeddings, allow_dangerous_deserialization=True)
                logger.info("Vector store initialized")
            else:
                logger.error(f"Index file not found at {index_file_path}")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")


def save_vector_store(vector_store):
    index_dir = "faiss_index_openai"
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)

    index_file_path = os.path.join(index_dir, "index.faiss")
    try:
        vector_store.save_local(index_file_path)
        logger.info(f"Vector store saved at {index_file_path}")
    except Exception as e:
        logger.error(f"Failed to save vector store: {e}")


def load_vector_store():
    global vector_store
    if vector_store is None:
        initialize_vector_store()


async def fetch(session, url):
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        logger.error(f"Error fetching website {url}: {e}")
        return None


async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        return await asyncio.gather(*tasks)


def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(raw_text)


def get_vector_store_genai(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    global vector_store
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    save_vector_store(vector_store)


def get_conversational_chain():
    prompt_template = """
    Answer the question in a friendly and conversational manner, providing as much detail as possible based on the provided context.
    Imagine you are a company representative and answer the user's question in a warm and engaging way, as if you enjoy talking and helping out. If the answer is not in the provided context, politely ask the user to rephrase or provide more information for clarity.
    Keep answers concise, around 30 to 40 words, and provide general information. Avoid presenting the content or data in a code format.\n\n
    Context:\n {context}\n
    Question:\n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    return LLMChain(llm=model, prompt=prompt)


def chat_with_context(context, question):
    greetings = ["hello", "hey", "good morning",
                 "good afternoon", "good evening", "what's up"]
    if any(greeting in question.lower() for greeting in greetings):
        return "Hi there! How can I help you today?"
    else:
        chain = get_conversational_chain()
        response = chain({"context": context, "question": question})
        return response['output_text']


@cached(cache)
def get_cached_response(user_question):
    logger.debug(f"Cache miss for question: {user_question}")
    load_vector_store()
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    context = "\n".join([doc.page_content for doc in docs[:10]])
    response = chain.run(context=context, question=user_question)
    return response


def handle_query(request):
    website_urls = [
        'https://sites.google.com/view/atharvainnovision/services?authuser=0',
        'https://sites.google.com/view/atharvainnovision/home?authuser=0',
        'https://sites.google.com/view/atharvainnovision/contact?authuser=0'
    ]

    raw_texts = asyncio.run(fetch_all(website_urls))
    all_text_chunks = []
    for raw_text in raw_texts:
        if raw_text:
            text_chunks = get_text_chunks(raw_text)
            all_text_chunks.extend(text_chunks)
        else:
            logger.error("Failed to fetch data from one of the websites")

    if all_text_chunks:
        get_vector_store_genai(all_text_chunks)
        if request.method == 'POST':
            data = json.loads(request.body.decode('utf-8'))
            user_query = data.get('message', '')
            if user_query:
                response = get_cached_response(user_query)
                return JsonResponse({'response': response})
            else:
                return JsonResponse({'error': 'Invalid request. Please provide a message in the request body.'})
    else:
        return JsonResponse({'error': 'Failed to fetch data from any of the websites.'})


# Call this function during server startup to initialize the vector store
initialize_vector_store()
