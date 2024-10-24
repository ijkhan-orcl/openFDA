import os
import json
import fitz
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langdetect import detect
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from pprint import pprint
from langgraph.graph import END, StateGraph
import streamlit as st
from langchain_openai import OpenAIEmbeddings
import openai
import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()


# Embeddings
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")


# Choose the LLM Model
llm = ChatGroq(temperature=0,
                model_name="Llama3-70b-8192",
                api_key="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")

# Tavily API
os.environ['TAVILY_API_KEY'] = "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"
web_search_tool = TavilySearchResults(k=3)

# Global counter
counter = 0

# Prompt for the translator
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a language translator for translating the text to English.
    Strictly return the only the translated text without any additional information.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Input Text: {input}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["input"],
)
translator = prompt | llm | StrOutputParser()

# def extract_json_content(json_path):
#     with open(json_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)
    
#     # Assuming the text you want is in a specific key, e.g., 'text'
#     # You may need to adjust this based on the structure of your JSON
#     text_content = data.get('text', '')  # Adjust 'text' to the relevant key in your JSON
#     return text_content
def extract_json_content(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    results = []
    data=data['results']
    for item in data:
        recall_detail = {
            "cfres_id": item.get("cfres_id"),
            "product_res_number": item.get("product_res_number"),
            "event_date_initiated": item.get("event_date_initiated"),
            "event_date_posted": item.get("event_date_posted"),
            "recall_status": item.get("recall_status"),
            "res_event_number": item.get("res_event_number"),
            "product_code": item.get("product_code"),
            "k_numbers": item.get("k_numbers", []),  # Assuming k_numbers can be a list
            "product_description": item.get("product_description"),
            "code_info": item.get("code_info"),
            "recalling_firm": item.get("recalling_firm"),
            "address": {
                "address_1": item.get("address_1"),
                "city": item.get("city"),
                "state": item.get("state"),
                "postal_code": item.get("postal_code")
            },
            "additional_info_contact": item.get("additional_info_contact"),
            "reason_for_recall": item.get("reason_for_recall"),
            "root_cause_description": item.get("root_cause_description"),
            "actions_required": {
                "customers": [
                    "Complete the Medical Facility Acknowledgement Form.",
                    "Remove the applicator from service and discard it if impacted.",
                    "Forward this notice to those who utilize the product.",
                    "Return the Medical Facility Acknowledgement Form via email or fax.",
                    "Keep a copy of the form for your records."
                ],
                "distributors": [
                    "Complete the Distributor Acknowledgement Form.",
                    "Include a copy of this notice with every shipment of impacted products.",
                    "Return the Acknowledgement Form via email or fax.",
                    "Keep a copy of the form for your records.",
                    "Forward a copy of this notice to customers who purchased the impacted products."
                ]
            },
            "product_quantity": item.get("product_quantity"),
            "distribution_pattern": item.get("distribution_pattern"),
            # "openfda": item.get("openfda", {})  # Assuming openfda can be a dict
        }
        results.append(recall_detail)
    return results

def initialize_vectorstore(file_path):
    
    long_text = extract_json_content(file_path)

    if isinstance(long_text, list):
        long_text = " /*/ ".join([json.dumps(item) for item in long_text])
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=900)
    chunks = splitter.split_text(long_text)

    # Initialize the embeddings model and Chroma DB
    vectorstore = Chroma(embedding_function=embed_model)

    # Translate and prepare chunks with IDs
    ids = []
    for i in range(len(chunks)):
        if detect(chunks[i]) != 'en':
            chunks[i] = translator.invoke({"input": chunks[i]})
        ids.append(f"chunk_{i}")  # Generate a unique ID for each chunk

    # Embed and store the chunks in Chroma DB
    if chunks:  # Check if there are any chunks
        vectorstore.add_texts(texts=chunks, ids=ids)

    # Initialize the retriever
    global retriever
    retriever = vectorstore.as_retriever()
    return vectorstore

# def initialize_vectorstore(file_path):
#     long_text = extract_json_content(file_path)

#     if isinstance(long_text, list):
#         long_text = " /*/ ".join([json.dumps(item) for item in long_text])
    
#     splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=900)
#     chunks = splitter.split_text(long_text)

#     # Initialize the embeddings model and Chroma DB
#     vectorstore = Chroma(embedding_function=embed_model)

#     # Translate and prepare chunks with IDs
#     ids = []
#     for i in range(len(chunks)):
#         if detect(chunks[i]) != 'en':
#             chunks[i] = translator.invoke({"input": chunks[i]})
#         ids.append(f"chunk_{i}")  # Generate a unique ID for each chunk

#     # Embed and store the chunks in Chroma DB
#     if chunks:  # Check if there are any chunks
#         # Here we avoid passing metadatas explicitly unless needed
#         vectorstore.add_texts(chunks, ids, metadatas=[None] * len(chunks))

#     # Initialize the retriever
#     global retriever
#     retriever = vectorstore.as_retriever()
#     return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_prompt_templates():

    question_router_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are responsible for deciding the best data source for FDA recall-related queries.
    For questions about product codes, batch numbers, device classifications, recall reasons, company names, code info, address, root cause, actions, product quantity, distribution pattern, route to 'vectorstore'.
    For all other types of queries, such as general FDA regulations or unrelated product inquiries, route to 'web_search'.
    Return a JSON with a single key 'datasource' indicating either 'web_search' or 'vectorstore'. 
    Question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
    )
    question_router = question_router_prompt | llm | JsonOutputParser()

    # Prompt for QA Assistant
    qa_assistant_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert assistant designed to provide concise and accurate answers
    to FDA-related questions, specifically regarding product recalls, device safety, product codes, batch numbers, device classifications, recall reasons, company names, code info, address, root cause, actions, product quantity, distribution pattern, regulations, and compliance. 
    Use the provided context to answer the question in three sentences or less. If the context doesn’t provide enough information, respond with "I don't know."
    Stay factual and focused.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"],
    )
    rag_chain = qa_assistant_prompt | llm | StrOutputParser()

    # Prompt for grading relevance of the retrieved documents
    retrieval_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are evaluating whether a retrieved document is relevant to the user's FDA-related question.
    Check if the document addresses key FDA topics, such as product recalls, device safety, product codes, batch numbers, device classifications, recall reasons, company names, code info, address, root cause, actions, product quantity, distribution pattern, supplier information, or related regulations. 
    If the document is related to these topics, score it 'yes'. If not, score it 'no'. 
    Provide a JSON with a single key 'score'. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Retrieved document: \n\n {document} \n\n
    User question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
    )   
    retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()

    # Check for hallucinations
    hallucination_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are assessing whether the provided answer is grounded in the supplied FDA-related facts.
    If the answer aligns with the given context on recalls, regulations, or related FDA data, score it as 'yes'. 
    If it diverges or introduces unsupported claims, score it as 'no'. 
    Provide a binary 'yes' or 'no' score in a JSON with a single key 'score'. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Supporting facts: \n ------- \n{documents}\n ------- \nAnswer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
    )
    hallucination_grader = hallucination_grader_prompt | llm | JsonOutputParser()

    # Grade the answer according to the question
    answer_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are assessing whether an answer effectively resolves an FDA-related question.
    Determine if the answer addresses key aspects such as product recalls, product codes, batch numbers, device classifications, recall reasons, company names, code info, address, root cause, actions, product quantity, distribution pattern, compliance, or regulations. 
    Provide a binary score 'yes' or 'no' as a JSON with a single key 'score'. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Answer:\n ------- \n{generation}\n ------- \nQuestion: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
    )
    answer_grader = answer_grader_prompt | llm | JsonOutputParser()
    return question_router,rag_chain,retrieval_grader, hallucination_grader, answer_grader

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
#
def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
#
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d})
        grade = score['score']
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
#
def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}

def irrelevant(state):
    """
    Web search based based on the question
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Appended web results to documents
    """
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents = [web_results]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"question": question, "generation": "The question seems not related to the document provided. The web results that were found are as followed: " + generation}

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source['datasource'])
    if source['datasource'] == 'web_search':
        print("---ASKED QUESTION IS NOT RELEVANT---")
        return "nodata"
    elif source['datasource'] == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")

    global counter

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score['score']

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score['score']
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            counter = counter + 1
        if counter < 3:
            return "not useful"
        else:
            return "useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        counter = counter + 1
        if counter < 3:
            return "not supported"
        else:
            return "useful"


def create_workflow(vectorstores, question_routers, rag_chains, retrieval_graders, hallucination_graders, answer_graders):
    from langgraph.graph import END, StateGraph
    from typing_extensions import TypedDict

    global question_router, rag_chain, retrieval_grader, hallucination_grader, answer_grader, vectorstore

    vectorstore = vectorstores
    question_router = question_routers
    rag_chain = rag_chains
    retrieval_grader = retrieval_graders
    hallucination_grader = hallucination_graders
    answer_grader = answer_graders

    ### State
    class GraphState(TypedDict):
        question : str
        generation : str
        web_search : str
        documents : List[str]

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("websearch", web_search) # web search
    workflow.add_node("retrieve", retrieve) # retrieve
    workflow.add_node("grade_documents", grade_documents) # grade documents
    workflow.add_node("generate", generate) # generatae
    workflow.add_node("irrelevant", irrelevant) # irrelevant

    workflow.set_conditional_entry_point(
        route_question,
        {
            "nodata": "irrelevant",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("irrelevant", "__end__")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )

    app = workflow.compile()
    return app