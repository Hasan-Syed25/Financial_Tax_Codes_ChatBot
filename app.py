import os
import json
import chainlit as cl
import torch
from transformers import AutoTokenizer
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core.ingestion import IngestionPipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank

# Set up the LLM
system_prompt = '''You are a Q&A assistant that helps with implementing appropriate taxes on US Products according to tax code in the following Format:\n{{\n"tax_code" : ##here comes the tax code\n}}. Your goal is to answer questions as accurately as possible based on the instructions and context provided.'''
query_wrapper_prompt = PromptTemplate('''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{query_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>''')

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

llm = HuggingFaceLLM(
    context_window=8192,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="auto",
    stopping_ids=stopping_ids,
    tokenizer_kwargs={"max_length": 4096},
    model_kwargs={"torch_dtype": torch.bfloat16}
)

# Set up the embedding model
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Configure Settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

# Set up the node parser
text_splitter = LangchainNodeParser(RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=100, chunk_overlap=0))

# Set up the reranker
rerank = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)

# Load documents
documents = SimpleDirectoryReader("./Data").load_data()

# Process documents
pipeline = IngestionPipeline(transformations=[text_splitter])
nodes = pipeline.run(documents=documents, in_place=True, show_progress=True)

# Create index
index = VectorStoreIndex(nodes)

@cl.on_chat_start
async def start():
    await cl.Message(
        author="Assistant",
        content="Hello! I'm an AI assistant specializing in US product tax codes. How may I help you?"
    ).send()
    # Create query engine
    query_engine = index.as_query_engine(similarity_top_k=10, node_postprocessors=[rerank])

    cl.user_session.set("query_engine", query_engine)

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")

    response = await cl.make_async(query_engine.query)(message.content)
    
    try:
        parsed_response = json.loads(response.response)
        formatted_response = f"Tax Code: {parsed_response['tax_code']}"
    except json.JSONDecodeError:
        formatted_response = response.response

    await cl.Message(content=formatted_response, author="Assistant").send()