from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# load embedding model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# load faiss index
vector_db = FAISS.load_local(
    "consumer_sentiment_faiss",
    embeddings,
    allow_dangerous_deserialization= True
)

# query="Top news related to mobile accesories"
# query="common complaints in beauty care products"
query="why home appliance having good reviews"

# apply similarity search
results = vector_db.similarity_search(query, k=10)


# dispaly result
retrived_documents=[]
for i, r in enumerate(results,1):
    # print(f"\nResult {i}")
    # print("Text", r.page_content)
    # print("metadata", r.metadata)
    # print("="*60)
    retrived_documents.append(r.page_content)

prompt=f"""

    You are a market intelligence analyst
    
    using only the information from the provided context
    
    give response based on the question
    
    do not use bullet points, headings, or sections
    do not add external knowledge
    
    Context:
    {retrived_documents}
    
    Question:
    {query}
    
    Answer:
"""   


# Google Genai
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv


load_dotenv()
client = genai.Client(api_key=os.getenv("Gemini_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=0   # disable thinking
        ),
        temperature=0.2
    )
)

print(response.text)


print("______________________________________________________________________")

# GROQ

import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.environ.get(os.getenv("GROQ_API_KEY")),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content":prompt,
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)

