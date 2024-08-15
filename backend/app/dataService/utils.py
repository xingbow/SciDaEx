"""
utils.py - Utility Functions for PDF Processing and RAG System

This module contains essential utility functions for processing PDF documents,
managing document vector stores, and implementing a Retrieval-Augmented Generation (RAG) system.
It provides tools for extracting and processing figures, tables, and meta-information from PDFs,
as well as functions for building and managing document embeddings and retrieval systems.

Key functionalities include:

1. PDF Content Extraction:
   - Extract figures, tables, and meta-information from PDF files
   - Process single PDFs or entire folders of PDFs
   - Support for both LLM-based and Adobe API-based extraction methods

2. Vector Store Management:
   - Build and save local document vector stores
   - Create multi-vector retrievers for efficient document retrieval

3. RAG System Components:
   - Build RAG chains for question answering
   - Manage document retrieval and context generation

Main Functions:
- process_figures: Process figures from multiple PDFs
- process_tables: Process tables from multiple PDFs
- process_meta_information: Process meta information from multiple PDFs
- process_single_pdf_figure: Process figures from a single PDF
- process_single_pdf_table: Process tables from a single PDF
- process_single_pdf_meta_information: Process meta information from a single PDF
- build_local_document_vector_store: Build a vector store from documents
- save_local_document_vector_store: Save a vector store to disk
- build_multivector_retriever: Create a multi-vector retriever
- build_rag_chain: Construct a RAG chain for question answering

"""
# Standard library imports
import ast
import base64
import csv
import io
import json
import os
import os.path
import pickle
import re
import shutil
import uuid
import zipfile
from io import StringIO
from operator import itemgetter
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import pandas as pd
import PyPDF2
import requests
import tiktoken
from PIL import Image
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# Adobe PDF Services SDK imports
from adobe.pdfservices.operation.auth.credentials import Credentials
from adobe.pdfservices.operation.execution_context import ExecutionContext
from adobe.pdfservices.operation.io.file_ref import FileRef
from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import ExtractPDFOptions
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_renditions_element_type import ExtractRenditionsElementType
from adobe.pdfservices.operation.pdfops.options.extractpdf.table_structure_type import TableStructureType

# Langchain imports
from langchain.pydantic_v1 import BaseModel, Field
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Local imports
try:
    import app.dataService.globalVariable as GV
    from app.dataService.globalVariable import (
        table_extract_prompt_template,
        table_structure_prompt_template,
        meta_info_extract_prompt_template,
        figure_describe_prompt_template,
        table_structure_prompt_templatev2
    )
except ImportError:
    import globalVariable as GV
    from globalVariable import (
        table_extract_prompt_template,
        table_structure_prompt_template,
        meta_info_extract_prompt_template,
        figure_describe_prompt_template,
        table_structure_prompt_templatev2
    )
    
def summarize_text_openai(texts: List[str], openai_key: str, model_name: str = "gpt-3.5-turbo-1106", max_workers: int = 5) -> list[dict[str, str]]:
    prompt_text = """You are an assistant tasked with summarizing tables and text. \
    Give a concise summary of the table or text. Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(temperature=0, model=model_name, openai_api_key=openai_key)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    
    results = []
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_text = {executor.submit(summarize_single_text, text, summarize_chain): text for text in texts}
        
        for future in tqdm(as_completed(future_to_text), total=len(texts), desc="Summarizing texts"):
            original_text = future_to_text[future]
            summary, error = future.result()
            results.append({"original": original_text, "summary": summary})
            if error:
                errors.append(error)

    if errors:
        print(f"Encountered {len(errors)} errors while summarizing texts.")

    return results

def summarize_single_text(text: str, summarize_chain) -> tuple[str, str]:
    try:
        summary = summarize_chain.invoke(text)
        return summary, None
    except Exception as e:
        error_message = f"Error summarizing text: {e}"
        print(error_message)
        return "Summary unavailable due to processing error.", error_message

def build_local_document_vector_store(texts: List[str], openai_key: str) -> tuple[FAISS, InMemoryStore]:
    id_key = "doc_id"
    
    # Summarize the texts
    summary_results = summarize_text_openai(texts, openai_key)
    
    # Create documents, filtering out failed summaries
    doc_ids = []
    summary_texts = []
    valid_texts = []
    
    for result in summary_results:
        if result["summary"] != "Summary unavailable due to processing error.":
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            summary_texts.append(Document(page_content=result["summary"], metadata={id_key: doc_id}))
            valid_texts.append(result["original"])
    
    # Create vectorstore
    vectorstore = FAISS.from_documents(summary_texts, OpenAIEmbeddings(openai_api_key=openai_key))
    
    # Create docstore
    docstore = InMemoryStore()
    docstore.mset(list(zip(doc_ids, valid_texts)))
    
    return vectorstore, docstore

def save_local_document_vector_store(texts: list[str], output_vectorstore_path: str, output_docstore_path: str, openai_key: str):   
    id_key = "doc_id"
    
    # Summarize the texts
    summary_results = summarize_text_openai(texts, openai_key)
    
    # Create documents, filtering out failed summaries
    doc_ids = []
    summary_texts = []
    valid_texts = []
    
    for result in summary_results:
        if result["summary"] != "Summary unavailable due to processing error.":
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            summary_texts.append(Document(page_content=result["summary"], metadata={id_key: doc_id}))
            valid_texts.append(result["original"])

    # Create vectorstore
    vectorstore = FAISS.from_documents(summary_texts, OpenAIEmbeddings(openai_api_key=openai_key))
    
    # Create docstore
    docstore = InMemoryStore()
    docstore.mset(list(zip(doc_ids, valid_texts)))

    # Save the vectorstore and docstore to disk
    vectorstore.save_local(output_vectorstore_path)

    with open(output_docstore_path, "wb") as f:
        pickle.dump(docstore, f)
def build_multivector_retriever(vectorstore, docstore, id_key="doc_id"):
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
    )
    return retriever


def cut_string_to_token_length(string: str, encoding_name: str = "cl100k_base", max_token_length: int = 16000) -> str:
    # Cuts a string to fit within a specified token length.
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(string)

    if len(tokens) > max_token_length:
        # Cut the tokens to the specified length
        tokens = tokens[:max_token_length]
        # Convert the tokens back to a string
        cut_string = encoding.decode(tokens)
        return cut_string
    else:
        return string

def build_rag_chain(retriever):
    """
    Build a Retrieval-Augmented Generation (RAG) chain using the provided retriever.

    This function creates a chain that retrieves relevant context using the retriever,
    then generates an answer based on the retrieved context and the input question.

    Args:
        retriever: The retriever object used to fetch relevant context.

    Returns:
        RunnableParallel: A chain that processes input questions and returns both the context and the generated answer.
    """
    # RAG prompt template
    template = """Answer the question based only on the following context, which can include text and tables. 
    Tell me the answer in JSON format, which looks like the structure: {ans_format}
    
    Values of exceptions in JSON should be "Empty".

    Context: {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(temperature=0, 
                    #    model="gpt-4-1106-preview", 
                       model="gpt-4o",
                       openai_api_key = GV.openai_key,
                       model_kwargs={
                           # "seed": 42,
                           "response_format": { "type": "json_object" }
                       })
    retreival_chain = (
        {
            "question": itemgetter("question"),
            "ans_format": itemgetter("ans_format")
        } | RunnablePassthrough.assign(context=(itemgetter("question") | retriever) )
    )

    # Building the main answer chain
    chain_with_sources = retreival_chain | RunnableParallel({
        "context": RunnablePassthrough() | itemgetter("context"), 
        "answer": RunnablePassthrough() | prompt | model | StrOutputParser()
    })
    return chain_with_sources

#####################################################################################
# functional function encapsulation
import re

def check_quotes(s):
    # Find the positions of all double quotes
    quote_positions = [m.start() for m in re.finditer(r'"', s)]
    # List of double quote positions that need to be replaced
    replace_positions = []

    for pos in quote_positions:
        # Search backwards for the first non-whitespace character
        prev_char = None
        for i in range(pos - 1, -1, -1):
            if not s[i].isspace():
                prev_char = s[i]
                break

        # Search forwards for the first non-whitespace character
        next_char = None
        for i in range(pos + 1, len(s)):
            if not s[i].isspace():
                next_char = s[i]
                break

        # Check if the replacement condition is met
        if not ((prev_char in ['{', ':', ','] if prev_char else False) or
                (next_char in ['}', ':', ','] if next_char else False)):
            replace_positions.append(pos)

    # Generate a new string, replacing the necessary double quotes with a space
    new_s = list(s)
    for pos in replace_positions:
        new_s[pos] = ' '

    return ''.join(new_s)

def parse_list_of_dict(table_l: list):
    # Function to initialize string
    # Input: unstructured table list
    # Output: structured table dict 
    # Define regular expression pattern to match dictionary literals
    pattern = r"{[^{}]+}"
    tables_obj = []
    for table_str in table_l:
        # Find all matches of dictionary literals in input string
        matches = re.findall(pattern, table_str)
        try:
            table = []
            for match in matches:
                sanitized_match = check_quotes(match.replace("'", '"').replace("\n", " ").replace("\0", ''))
                try:
                    table.append(ast.literal_eval(sanitized_match))
                except Exception as e:
                    print(f"Failed to parse: {sanitized_match}")
                    print(f"Error: {e}")
                    continue
            tables_obj.extend(table)
        except Exception as e:
            raise Exception(f"Error parsing table string: {table_str}\n{e}")
    return tables_obj

def csv2html(csvreader):
    header = []
    data = []
    for i, row in enumerate(csvreader):
        if i == 0:
            header = row
        else:
            if len(row) == len(header):
                data.append(row)
    df = pd.DataFrame(data, columns=header)
    # html_table = df.to_html(index=False)
    json_table = df.to_json(orient="records")
    return json_table

def read_pdf(pdf_path:str, num_pages:int=None)->str:
    # function: read pdf
    pdf_file = open(pdf_path, 'rb')
    paper_content = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    total_pages = len(pdf_reader.pages)
    if num_pages is None:
        num_pages = total_pages
    for page_num in range(min(num_pages, total_pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        paper_content += text

    return paper_content

def get_pdf_page_count(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
        return num_pages
    except Exception as e:
        print(f"Error reading PDF file {pdf_path}: {str(e)}")
        return None

def split_pdf(pdf_path, chunk_size=10):
    reader = PyPDF2.PdfReader(pdf_path)
    total_pages = len(reader.pages)
    chunks = []

    for start in range(0, total_pages, chunk_size):
        end = min(start + chunk_size, total_pages)
        output = PyPDF2.PdfWriter()
        for page in range(start, end):
            output.add_page(reader.pages[page])
        
        chunk_name = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_chunk_{start+1}-{end}.pdf"
        chunk_path = os.path.join(os.path.dirname(pdf_path), chunk_name)
        with open(chunk_path, "wb") as output_stream:
            output.write(output_stream)
        chunks.append(chunk_path)

    return chunks
def extract_sentences_with_keywords(pdf_path, keyword_list, mode=0):
    # extract the sentences including the keywords
    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    # mode = 0: table, mode = 1: figure
    sentences_dict = {keyword: [] for keyword in keyword_list}
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.getPage(page_num)
        text = page.extract_text()
        text_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z][a-z])|(?<=[.!?])\s+(?=[A-Z]{2,})', text)
        # print(text_sentences)
        # for t in text_sentences:
        #     print(t)
        #     print('----------------------------------------\n')
        sentence_count = 0
        for sentence in text_sentences:
            for keyword in keyword_list:
                try:
                    number = keyword.split(' ')[1]
                    keyword_pattern = ''
                    if mode == 0:
                        # keyword_pattern = fr"Table\s*/?\s*(?:\\u\w{4}|\S+)?\s*{re.escape(number)}"
                        # keyword_pattern = fr"Table\s*/?\s*(?:\\u\w{4}|\S+)\s*{re.escape(number)}"
                        keyword_pattern = fr"(?i)Table\s*/?\s*(?:\\u\w{4}|\S+)?\s*{re.escape(number)}"
                    if mode == 1:
                        keyword_pattern = fr"(?i)(?:figure|fig\.)\s+{re.escape(number)}"
                    match = re.search(keyword_pattern, sentence)
                except:
                    print("Error: ", keyword)
                    match = None
                if match:
                    start_index = match.end()
                    if re.search(r"\S", sentence[start_index:]) != None:
                        next_non_empty_char = re.search(r"\S", sentence[start_index:]).group()
                        if not next_non_empty_char.isupper():
                            sentence_info = {
                                "page_number": page_num,
                                "sentence_number": sentence_count,
                                "sentence_content": sentence
                            }
                            sentences_dict[keyword].append(sentence_info)
            sentence_count += 1
    return sentences_dict

def combine_results(all_results):
    combined = {"elements": []}
    page_offset = 0

    for chunk_result in all_results:
        for element in chunk_result["elements"]:
            if "Page" in element:
                element["Page"] += page_offset
            combined["elements"].append(element)
        
        # Update page offset for the next chunk
        max_page = max([elem["Page"] for elem in chunk_result["elements"] if "Page" in elem], default=0)
        page_offset += max_page

    return combined

def extract_figures_tables_through_adobe(pdf_path=os.path.join(GV.data_dir, "Abiodun.pdf"), 
                                         client_id=GV.adobe_client_id, 
                                         client_secret=GV.adobe_client_secret):
    # extract pdf figures
    pdf_name = pdf_path.split('/')[-1].split('.')[0]
    zip_file = os.path.join(GV.temp_dir, pdf_name + ".zip")
    output_folder = os.path.join(GV.temp_dir, pdf_name)
    
    if os.path.exists(output_folder):
        return "This pdf has been processed."
    
    #Initial setup, create credentials instance.
    credentials = Credentials.service_principal_credentials_builder().with_client_id(client_id).with_client_secret(client_secret).build()

    #Create an ExecutionContext using credentials and create a new operation instance.
    execution_context = ExecutionContext.create(credentials)
    extract_pdf_operation = ExtractPDFOperation.create_new()

    #Set operation input from a source file.
    source = FileRef.create_from_local_file(pdf_path)
    extract_pdf_operation.set_input(source)

    #Build ExtractPDF options and set them into the operation
    # new version
    extract_pdf_options: ExtractPDFOptions = ExtractPDFOptions.builder() \
        .with_elements_to_extract([ExtractElementType.TEXT, ExtractElementType.TABLES]) \
        .with_elements_to_extract_renditions([ExtractRenditionsElementType.TABLES,ExtractRenditionsElementType.FIGURES]) \
        .build()
    extract_pdf_operation.set_options(extract_pdf_options)
    #Execute the operation.
    result: FileRef = extract_pdf_operation.execute(execution_context)

    #Save the result to the specified location.
    result.save_as(zip_file)

    # unzip and remove ZIP
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    os.remove(zip_file)

#####################################################################################
# Figure special functions
def normalize_figure_name(figure_name):
    # finction: normalize the figure name, such as Fig. 1, FIG. 1, FIGURE 1 -> Figure 1
    pattern = r'^(.*?)(\d+)'
    match = re.search(pattern, figure_name)
    
    if match:
        figure_number = match.group(2)
        normalized_name = f"Figure {figure_number}"
        return normalized_name
    
    return figure_name

def check_merge_condition_position(bounds1, bounds2, threshold):
    # function: judge whether the figures should be mergred based on its x1, y1, x2, y2
    if abs(bounds1[0] - bounds2[0]) <= threshold and abs(bounds1[2] - bounds2[2]) <= threshold:
        if min(abs(bounds1[1] - bounds2[1]), abs(bounds1[1] - bounds2[3]), abs(bounds1[3] - bounds2[1]), abs(bounds1[3] - bounds2[3])) < 150:
            return True
    if abs(bounds1[1] - bounds2[1]) <= threshold and abs(bounds1[3] - bounds2[3]) <= threshold:
        if min(abs(bounds1[0] - bounds2[0]), abs(bounds1[0] - bounds2[2]), abs(bounds1[2] - bounds2[0]), abs(bounds1[2] - bounds2[2])) < 150:
            return True
    return False

def split_text_to_extract_number(sentence):
    # function: obtain the figure number in the caption
    pattern = r'^(.*?)(\d+)'
    match = re.search(pattern, sentence)
    return match.group(0)

def check_merge_condition_connection(index1, index2, data):
    # function: judge whether there is a figure caption between the index
    pattern = r'^(?i)\s*(F\s*I\s*G(\s*U\s*R\s*E)?)'
    for i in range(index1 + 1, index2):
        if 'Text' in data[i] and re.match(pattern, data[i]['Text']):
            figure_info = data[i]["Text"]
            figure_name = split_text_to_extract_number(figure_info)
            figure_info = figure_info[len(figure_name):]
            first_letter_index = next((index for index, char in enumerate(figure_info) if char.isalpha()), None)
            figure_info = figure_info[first_letter_index:]
            if figure_info[0].isupper():
                # figure caption exists, should not be merged
                return False
    return True

def extract_pdf_figures_page(pdf_path=os.path.join(GV.data_dir, "Abiodun.pdf"), page=4):
    # function: extract the figure for a specific page
    
    # count represents the figrue saving number
    pdf_name = pdf_path.split('/')[-1].split('.')[0]
    output_folder = os.path.join(GV.temp_dir, pdf_name)
    # output_folder = "app/figures_temp/" + pdf_name

    # read structuredData.json
    structured_data_file = os.path.join(output_folder, "structuredData.json")

    with open(structured_data_file, 'r') as f:
        structured_data = json.load(f)["elements"]
        filtered_data_pair = [(index, element) for index, element in enumerate(structured_data) if "filePaths" in element and element["filePaths"][0].endswith(".png") and element["Page"] == page]
        index_data = [item[0] for item in filtered_data_pair]
        filtered_data = [item[1] for item in filtered_data_pair]
        merged_groups = []
        for i in range(len(filtered_data)):
            if any(filtered_data[i] in group for group in merged_groups):
                continue
            group = [filtered_data[i]["filePaths"]]
            for j in range(i + 1, len(filtered_data)):
                if any(filtered_data[j]["filePaths"] in group for group in merged_groups):
                    continue
                if check_merge_condition_position(filtered_data[i]["Bounds"], filtered_data[j]["Bounds"], 15) and check_merge_condition_connection(index_data[i], index_data[j], structured_data):
                    group.append(filtered_data[j]["filePaths"])
            if len(group) > 1:
                merged_groups.append(group)

        has_judged = []
        figure_list = []
        for fd in filtered_data:
            if fd["filePaths"] in has_judged:
                continue
            current_point = fd["filePaths"]
            single = True
            for m in merged_groups:
                if current_point in m:
                    figure_list.append(m)
                    has_judged += m
                    single = False
            if single == True:
                figure_list.append(fd["filePaths"])
                has_judged.append(fd["filePaths"])
        return figure_list

def combine_figures(image_files, save_path):
    # function: combine the figures which should be a whole figure
    images = [Image.open(image_file) for image_file in image_files]
    images = [Image.open(image_file) for image_file in image_files]

    min_width = min(image.width for image in images)
    min_height = min(image.height for image in images)

    target_width = min_width
    target_height = min_height * len(images)

    target_image = Image.new("RGBA", (target_width, target_height))

    y_offset = 0
    for image in images:
        width, height = image.size
        x_offset = (target_width - width) // 2
        target_image.paste(image, (x_offset, y_offset))
        y_offset += height

    target_image.save(save_path)

def merge_figures(figure_list):
    # function: if there is still repeated figures, merge them
    merged_figures = {}
    
    for figure in figure_list:
        figure_name = figure['figure_name']
        figure_caption = figure['figure_caption']
        
        if figure_name in merged_figures:
            merged_figures[figure_name].append(figure_caption)
        else:
            merged_figures[figure_name] = [figure_caption]
    
    sorted_figures = sorted(merged_figures.items(), key=lambda x: int(x[0].split(' ')[-1]))
    
    merged_captions = []
    for figure_name, captions in sorted_figures:
        caption_text = ' '.join(captions)
        merged_captions.append({'figure_name': figure_name, 'figure_caption': caption_text})
    
    return merged_captions


def extract_figure_caption_and_page_adobe(pdf_path):
    # function: leverage adobe api to extract the figure captions (* used at this time)
   
    # count represents the figrue saving number
    pdf_name = pdf_path.split('/')[-1].split('.')[0]
    output_folder = os.path.join(GV.temp_dir, pdf_name)
    # output_folder = "app/figures_temp/" + pdf_name

    # read structuredData.json
    structured_data_file = os.path.join(output_folder, "structuredData.json")
    
    pattern = r'^(?i)\s*(F\s*I\s*G(\s*U\s*R\s*E)?)'

    page_list = []
    figure_list = []
    with open(structured_data_file, 'r') as f:
        structured_data = json.load(f)["elements"]
        
        for json_data in structured_data:
            if 'Text' in json_data and re.match(pattern, json_data['Text']):
                if json_data["Page"] not in page_list:
                    page_list.append(json_data["Page"])
                figure_info = json_data["Text"]
                try:
                    figure_name = split_text_to_extract_number(figure_info)
                except:
                    figure_name = figure_info

                figure_info = figure_info[len(figure_name):]
                first_letter_index = next((index for index, char in enumerate(figure_info) if char.isalpha()), None)
                figure_info = figure_info[first_letter_index:]
                
                if figure_info and figure_info[0].isupper():
                    figure_list.append({
                        "figure_name": normalize_figure_name(figure_name),
                        "figure_caption": figure_info,
                        "figure_content": "none", 
                    })
    return figure_list, page_list

def extract_pdf_figure(pdf_path):
    # 1. need to set the output dir at first
    output_dir = os.path.join(GV.data_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pdf_name = pdf_path.split('/')[-1].split('.')[0]
    
    # 2. use adobe api to extract all the figures
    extract_figures_tables_through_adobe(pdf_path)

    # 3. use adobe to extract the pages with figures and the figure caption
    figure_info_list, page_included_figure = extract_figure_caption_and_page_adobe(pdf_path)

    # 4. search the corresponding images
    save_number = 1
    for p in page_included_figure:
        figure_extracted = extract_pdf_figures_page(pdf_path, p)
        # start to save the figure
        for f in figure_extracted:
            save_path = os.path.join(output_dir, pdf_name + '_' + str(save_number) + '.png')
            if isinstance(f[0], list):
                # need to combine several figures
                figure_path_list_temp = []
                for ff in f:
                    figure_path_list_temp.append(os.path.join(GV.temp_dir, pdf_name + '/' + ff[0]))
                    # figure_path_list_temp.append('app/figures_temp/' + pdf_name + '/' + ff[0])  
                combine_figures(figure_path_list_temp, save_path)
            else:
                # this is the final figure
                shutil.copy2(os.path.join(GV.temp_dir, pdf_name + '/' + f[0]), save_path)
                # shutil.copy2('figures_temp/' + pdf_name + '/' + f[0], save_path)
            if (save_number < len(figure_info_list) + 1):
                figure_info_list[save_number - 1]['figure_url'] = save_path
            save_number += 1

    # 4. extract the mentioned sentences
    figure_name_list = []
    for figure_inf in figure_info_list:
        figure_name_list.append(figure_inf['figure_name'])
        # figure_name_list: ["Figure 1", "Figure 2", ...]
    figure_sentences = extract_sentences_with_keywords(pdf_path, figure_name_list, 1)

    # - add the sentences into the tables
    for figure_info in figure_info_list:
        figure_info['figure_mentioned'] = figure_sentences[figure_info['figure_name']]

    return figure_info_list

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def describe_figure(image_path, caption, model, api_key):
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": model,
    "messages": figure_describe_prompt_template(caption, base64_image),
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()["choices"][0]["message"]["content"]

def process_figures(pdf_fold, figure_fold, model, openai_api_key):
    """
    Process figures from all PDFs in a folder and save the results as JSON files.

    Args:
        pdf_fold (str): Path to the folder containing PDF files.
        figure_fold (str): Path to the folder where figure JSON files will be saved.
        model (str): Name of the LLM model to use for figure description, such as "gpt-4o" or "gpt-4-turbo"
        openai_api_key (str): OpenAI API key for LLM access.
    """
    print("figure extraction")
    for pdf_file in tqdm(os.listdir(pdf_fold)):
        print(pdf_file)
        if pdf_file[-3:] != 'pdf':
            continue
        pdf_name = pdf_file.split(".")[0]
        pdf_path = os.path.join(pdf_fold, pdf_file)
        try:
            figure_example = extract_pdf_figure(pdf_path)
            # leverage gpt-4v to read figure_url and obtain the figure_content
            # get the figure to generate the answer
            for i in range(0, len(figure_example)):
                if "figure_url" not in figure_example[i] or "figure_caption" not in figure_example[i]:
                    continue
                caption = figure_example[i]["figure_caption"]
                if model == "none":
                    response = ""
                else:
                    response = describe_figure(figure_example[i]["figure_url"], caption, model, openai_api_key)
                figure_example[i]["figure_content"] = response
            with open(os.path.join(figure_fold, pdf_name + ".json"), "w") as f:
                json.dump(figure_example, f)
        except Exception as e:
            print(f"Error processing PDF file {pdf_file}: {str(e)}")
            figure_example = []
        with open(os.path.join(figure_fold, pdf_name + ".json"), "w") as f:
            json.dump(figure_example, f)
            
#####################################################################################
# Single PDF figure special functions
def process_single_pdf_figure(pdf_path, figure_fold, model, openai_api_key):
    """
    Process figures from a single PDF file and save the results as a JSON file.

    Args:
        pdf_path (str): Path to the PDF file.
        figure_fold (str): Path to the folder where the figure JSON file will be saved.
        model (str): Name of the LLM model to use for figure description, such as "gpt-4o", "gpt-4-turbo"
        openai_api_key (str): OpenAI API key for LLM access.
    """
    print("Processing single PDF for figure extraction")
    if os.path.basename(pdf_path)[-3:] != 'pdf':
        raise Exception("Invalid PDF file")
        return
    pdf_name = os.path.basename(pdf_path).split(".")[0]
    try:
        figure_example = extract_pdf_figure(pdf_path)
    except Exception as e:
        print(f"Error processing PDF file {pdf_path}: {str(e)}")
        figure_example = []
    for i in range(0, len(figure_example)):
        if "figure_url" not in figure_example[i] or "figure_caption" not in figure_example[i]:
            continue
        caption = figure_example[i]["figure_caption"]
        if model == "none":
            response = ""
        else:
            response = describe_figure(figure_example[i]["figure_url"], caption, model, openai_api_key)
        figure_example[i]["figure_content"] = response
    with open(os.path.join(figure_fold, pdf_name + ".json"), "w") as f:
        json.dump(figure_example, f)
#####################################################################################
# Table special functions
def parse_table_content(table_content):
    # table_content formation will be 
    # ｜(1) Table caption:
    # ｜Bioactive components of fresh orange hybrid maize at different harvesting time across genotypes and locations
    # ｜
    # ｜(2) Table content in CSV format:
    # ｜"Harvesting time","Bioactive properties","Lutein (µ/g)","Zeaxanthin (µ/g)","β-Cryptoxanthin (µ/g)","Phytate (%)","Tannin (%)","Vitamin C (mg/100g)"

    # 1. parse table_caption & table_content
    table_content_list = table_content.split('(1) Table caption:')[1].split('(2) Table content in CSV format:')
    table_caption = table_content_list[0].strip()
    table_content_csv = table_content_list[1].strip()
    # 2. parse table_content_csv
    reader = csv.reader(io.StringIO(table_content_csv))
    table_content_csv = reader
    return [table_caption, table_content_csv]

def match_lists(list1, list2):
    # function: match the table caption with the table content
    result1 = []
    result2 = []

    for num2 in list2:
        min_diff = float('inf')
        closest_num = None

        for num1 in list1:
            diff = abs(num2 - num1)
            if diff < min_diff:
                min_diff = diff
                closest_num = num1

        result1.append(closest_num)
        result2.append(num2)
        
        if len(list2) < len(list1) and closest_num != None:
            list1.remove(closest_num)

    return [result1, result2]

def normalize_table_name(figure_name):
    # normalize the table name: Table 1, Table 2, ...
    pattern = r'^(.*?)(\d+)'
    match = re.search(pattern, figure_name)
    
    if match:
        figure_number = match.group(2)
        normalized_name = f"Table {figure_number}"
        return normalized_name
    
    return figure_name

def extract_pdf_table_llm_new(pdf_path, model_name, openai_api_key):
    # table structure model
    structure_model = ChatOpenAI(model_name=model_name, 
                                 temperature=0, 
                                 openai_api_key = openai_api_key, 
                                 model_kwargs={"response_format": { "type": "json_object" }})
    # general model
    model = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key = openai_api_key)
    table_extract_prompt = PromptTemplate(
        template = table_extract_prompt_template,
        input_variables=["page_content"]
    )
    table_extract_chain = table_extract_prompt | model
    # read pdf
    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    # extract tables
    Table = []
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.getPage(page_num)
        text = page.extract_text()
        table = table_extract_chain.invoke({"page_content": text}).content
        # print(f"page: {page_num}: {table}")
        # if "no" not in list(json.loads(table).values()) and "No" not in list(json.loads(table).values()):
        if table != "no" and table != "No":
            Table.append(table)
    table_inf_list = parse_list_of_dict(Table) # TODO: can consider use SimpleJsonOutputParser later
    # print(f"table_inf_list: {table_inf_list}")
    # extract the mentioned sentences
    table_name_list = []
    
    for table_inf in table_inf_list:
        # Ensure table_inf is a dictionary
        if isinstance(table_inf, dict):
            table_name_list.append(table_inf['table_name'])
        else:
            raise TypeError(f"Expected a dictionary, got {type(table_inf)}")
        # table_name_list.append(table_inf['table_name'])
        # # table_name_list: ["Table 1", "Table 2", ...]
    table_sentences = extract_sentences_with_keywords(pdf_path, table_name_list)

    # add the sentences into the tables
    valid_table_inf_list = []
    for table_inf in table_inf_list:
        table_inf['table_mentioned'] = table_sentences[table_inf['table_name']]
        # new version
        response = structure_model.invoke(table_structure_prompt_templatev2.format(table_information = table_inf['table_content'])).content
        try:
            info = json.loads(response)
            table_inf['table_caption'] = info.get("table_caption", "No caption")
            table_inf['table_content'] = pd.read_csv(io.StringIO(info["table_content"]), on_bad_lines="skip").to_json(orient="records")
            valid_table_inf_list.append(table_inf)  # Add only valid table_inf to the list
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response: {response}")
            continue
        except Exception as e:
            print(f"Error processing table information: {e}")
            continue
    pdf_file.close()
    return valid_table_inf_list

def extract_pdf_table_llm(pdf_path, model, openai_api_key):
    # function: extract odf tables through llm
    os.environ["OPENAI_API_KEY"] = openai_api_key
    model = ChatOpenAI(model_name="gpt-4-1106-preview", 
                       temperature=0, 
                       openai_api_key = openai_api_key)

    table_extract_prompt = PromptTemplate(
        template = table_extract_prompt_template,
        input_variables=["page_content"]
    )
    table_structure_prompt = PromptTemplate(
        template = table_structure_prompt_template,
        input_variables=["table_information"]
    )
    # json_parser = SimpleJsonOutputParser()

    table_extract_chain = table_extract_prompt | model
    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)

    print("===table extraction===")
    # extract tables
    Table = []
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        text = page.extract_text()
        # table = model.predict(table_extract_prompt.format(page_content=text))
        table = table_extract_chain.invoke({"page_content": text}).content
        print(table)
        if table != "no" and table != "No":
            Table.append(table)
    table_inf_list = parse_list_of_dict(Table) # TODO: can consider use SimpleJsonOutputParser later
    
    # extract the mentioned sentences
    table_name_list = []
    for table_inf in table_inf_list:
        table_name_list.append(table_inf['table_name'])
        # table_name_list: ["Table 1", "Table 2", ...]
    table_sentences = extract_sentences_with_keywords(pdf_path, table_name_list)
    
    # add the sentences into the tables
    for table_inf in table_inf_list:
        table_inf['table_mentioned'] = table_sentences[table_inf['table_name']]
        table_prompt_input = table_structure_prompt.format(table_information = table_inf['table_content'])
        # table_inf['table_content'] = model.predict(table_prompt_input)
        table_inf['table_content'] = model.invoke(table_prompt_input).content
        # print('----------\n')
        # print(table_inf['table_content'])
        [table_caption, table_content_csv] = parse_table_content(table_inf['table_content'])
        table_inf['table_caption'] = table_caption
        # csv
        table_inf['table_content'] = table_content_csv
        # HTML
        table_inf['table_content'] = csv2html(table_content_csv)
    # for table_inf in table_inf_list:
    #     print(table_inf)
    pdf_file.close()
    return table_inf_list

def extract_pdf_table_adobe(pdf_path):
    # exract tables from pdf through Adobe
    output_dir = os.path.join(GV.data_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pdf_name = pdf_path.split('/')[-1].split('.')[0]
    
    # 2. use adobe api to extract all the figures and tables
    extract_figures_tables_through_adobe(pdf_path)

    # 3. read the figures_temp/.../structuredData.json
    structured_data_file = os.path.join(GV.temp_dir, pdf_name, "structuredData.json")

    # 4. identify the table: fileoutpart[d].xlsx and identify the table caption
    with open(structured_data_file, 'r') as f:
        structured_data = json.load(f)["elements"]
        pattern = r"^\s*t\s*a\s*b\s*l\s*e\s*\d+\b"

        filtered_data_pair = [(index, element) for index, element in enumerate(structured_data) if "filePaths" in element and element["filePaths"][0].startswith("tables/fileoutpart")]
        index_table = [item[0] for item in filtered_data_pair]
        
        filtered_data_pair2 = [(index, element) for index, element in enumerate(structured_data) if "Text" in element and re.match(pattern, element["Text"], re.IGNORECASE)]
        index_caption = [item[0] for item in filtered_data_pair2]

    # 5. check whether there is a table caption after each element
    if len(index_table) != len(index_caption):
        index_table, index_caption = match_lists(index_table, index_caption)

    # 6. retrieve the content
    table_info_list = []
    table_name_list = []
    for i in range(len(index_table)):
        xlsx_path = structured_data[index_table[i]]["filePaths"][0]
        csv_table = pd.read_excel(os.path.join(GV.temp_dir, pdf_name, xlsx_path))
        table_info = structured_data[index_caption[i]]["Text"]
        table_name = normalize_table_name(split_text_to_extract_number(table_info))
        table_info = table_info[len(table_name):]
        table_letter_index = next((index for index, char in enumerate(table_info) if char.isalpha()), None)
        table_sentences = extract_sentences_with_keywords(pdf_path, [table_name])
        if table_name not in table_name_list:
            table_name_list.append(table_name)
            table_info_list.append({
                'table_name': table_name,
                'table_caption': table_info[table_letter_index:],
                'table_content': csv_table,
                'table_mentioned': table_sentences[table_name]
            })
        else:
            # combine the two table
            same_table = table_info_list[table_name_list.index(table_name)]
            for index, row in csv_table.iterrows():
                try:
                    if index != 0:
                        same_table['table_content'].loc[len(same_table['table_content'])] = row.values
                except:
                    continue
            # same_table['table_content'].to_excel('output.xlsx', index=False)
    # 7. transit to html/csv
    for i in range(len(table_info_list)):
        # table_info_list[i]['table_content'] = table_info_list[i]['table_content'].to_html()
        df = table_info_list[i]['table_content']
        # Removing '_x000D_' from column names
        df.columns = df.columns.str.replace('_x000D_', '')
        # Removing '_x000D_' from rows
        df = df.replace('_x000D_', '', regex=True)
        table_info_list[i]['table_content'] = df.to_json(orient="records")
    return table_info_list

def process_tables(pdf_folder, table_folder, model, openai_api_key):
    """
    Process tables from all PDFs in a folder and save the results as JSON files.

    Args:
        pdf_folder (str): Path to the folder containing PDF files.
        table_folder (str): Path to the folder where table JSON files will be saved.
        model (str): Name of the LLM model to use for table extraction.
        openai_api_key (str): OpenAI API key for LLM access.
    """
    for pidx, pdf_file in tqdm(enumerate(os.listdir(pdf_folder)), total=len(os.listdir(pdf_folder))):
        if pdf_file[-3:] != 'pdf':
            continue
        pdf_name = pdf_file.split(".")[0]
        pdf_path = os.path.join(pdf_folder, pdf_file)
        try:
            if model == "none":
                table_example = extract_pdf_table_adobe(pdf_path)
            else:
                table_example = extract_pdf_table_llm_new(pdf_path, model, openai_api_key)  
        except Exception as e:
            print(f"Error processing tables in PDF file {pdf_path}: {str(e)}")
            table_example = []
            
        with open(os.path.join(table_folder, pdf_name + ".json"), "w") as f:
            json.dump(table_example, f)

def process_single_pdf_table(pdf_path, table_folder, model, openai_api_key):
    """
    Process tables from a single PDF file and save the results as a JSON file.

    Args:
        pdf_path (str): Path to the PDF file.
        table_folder (str): Path to the folder where the table JSON file will be saved.
        model (str): Name of the LLM model to use for table extraction, such as "gpt-4o", "gpt-4-turbo"
        openai_api_key (str): OpenAI API key for LLM access.
    """
    if os.path.basename(pdf_path)[-3:] != 'pdf':
        raise Exception("Invalid PDF file")
    print("Processing single PDF for table extraction")
    pdf_name = os.path.basename(pdf_path).split(".")[0]
    try:
        if model == "none":
            table_example = extract_pdf_table_adobe(pdf_path)
        else:
            table_example = extract_pdf_table_llm_new(pdf_path, model, openai_api_key)
    except Exception as e:
        print(f"Error processing tables in PDF file {pdf_path}: {str(e)}")
        table_example = []
    with open(os.path.join(table_folder, pdf_name + ".json"), "w") as f:
        json.dump(table_example, f)

#####################################################################################
# meta-information special functions
def extract_pdf_meta_information(pdf_path, model, openai_api_key):
    # # ---use pdfminer---
    # print(pdf_path)
    os.environ["OPENAI_API_KEY"] = openai_api_key
    model = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0, openai_api_key = openai_api_key)
    paper_content = read_pdf(pdf_path, 2)
    # print("Start metainformation extraction")
    # print(paper_content)
    class Paper(BaseModel):
        Title: str = Field(description="the title of this paper")
        Author: str = Field(description="the names of authors in this paper")
        Abstract: str = Field(description="the abstract of this paper")
        Year: str = Field(description="the year this article was published")
        Journal_or_Conference: str = Field(description="the journal or conference of this paper")
        ISSN: str = Field(description="the ISSN of this paper")
        Volume: str = Field(description="the volume of this paper")
        Issue: str = Field(description="the issue of this paper")
        Page: str = Field(description="the pages of this paper")
        DOI: str = Field(description="the doi of this paper")
        Link: str = Field(description="the link of this paper, such as https//doi.org/10.2323/324323-f324ff.")
        Publisher: str = Field(description="the publisher of this paper")
        Language: str = Field(description="the language of this paper, such as English, Chinese")
    
    parser = PydanticOutputParser(pydantic_object = Paper)

    meta_extract_prompt = PromptTemplate(
        template = meta_info_extract_prompt_template,
        input_variables=["paper"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        )
    meta_extract_chain = meta_extract_prompt | model

    meta_info = meta_extract_chain.invoke({"paper": paper_content}).content

    meta_info = parser.parse(meta_info).__dict__

    if meta_info["Page"] != "none":
        # only one page
        if '-' not in meta_info["Page"]:
            meta_info["Page"] = {"start": meta_info["Page"], "end": meta_info["Page"]}
        # several pages
        else:
            page_range = meta_info["Page"].split('-')
            meta_info["Page"] = {"start": page_range[0], "end": page_range[1]}
    
    return meta_info

def process_meta_information(pdf_folder, meta_folder, model, openai_api_key):
    """
    Process meta information from all PDFs in a folder and save the results as JSON files.

    Args:
        pdf_folder (str): Path to the folder containing PDF files.
        meta_folder (str): Path to the folder where meta information JSON files will be saved.
        model (str): Name of the LLM model to use for meta information extraction.
        openai_api_key (str): OpenAI API key for LLM access.
    """
    for pidx, pdf_file in tqdm(enumerate(os.listdir(pdf_folder)), total=len(os.listdir(pdf_folder))):
        if pdf_file[-3:] != 'pdf':
            continue
        pdf_name = pdf_file.split(".")[0]
        pdf_path = os.path.join(pdf_folder, pdf_file)
        meta_example = extract_pdf_meta_information(pdf_path, model, openai_api_key)
        with open(os.path.join(meta_folder, pdf_name + ".json"), "w") as f:
            json.dump(meta_example, f)

def process_single_pdf_meta_information(pdf_path, meta_folder, model, openai_api_key):
    # Function to process meta information in a single PDF and save meta json file
    # input: pdf_path, meta_folder, model = "gpt-3.5-turbo-1106", openai_api_key
    if os.path.basename(pdf_path)[-3:] != 'pdf':
        raise Exception("Invalid PDF file")
    print("Processing single PDF for meta information extraction")
    pdf_name = os.path.basename(pdf_path).split(".")[0]
    meta_example = extract_pdf_meta_information(pdf_path, model, openai_api_key)
    with open(os.path.join(meta_folder, pdf_name + ".json"), "w") as f:
        json.dump(meta_example, f)
