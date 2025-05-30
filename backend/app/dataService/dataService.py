""""
dataService.py - Data Service for RAG-based Question Answering System

Description:
This module implements a data service for a Retrieval-Augmented Generation (RAG) based 
question answering system. It provides functionality for loading and managing document 
vector stores, processing PDF documents, and executing RAG-based question answering.

DataService Class:
   - Manages the loading and initialization of document vector stores
   - Handles PDF processing, including extraction of text, tables, and figures
   - Implements RAG-based question answering functionality

Main methods of DataService class:
- __init__: Initialize the DataService, loading necessary vector stores
- _load_vectorstores_: Load or create vector stores for document retrieval
- run_rag_qa: Execute RAG-based question answering on multiple PDF files
- process_rag_retriever: Process a single retriever for RAG-based QA

Main Components:
- DataService class: The core class that orchestrates the RAG process.
- Vector Store Loading: Efficient loading and management of precomputed vector stores.
- RAG Chain: Implementation of the Retrieval Augmented Generation chain for question answering.
- Parallel Batch Processing: Efficient handling of multiple PDF files concurrently.
"""

import os
import sys
import json
import openai
import numpy as np
import pandas as pd
import pickle
import uuid
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

try:
    import globalVariable as GV
    import utils
    import preprocess as preprocess
    import llm_eval as llmeval    
except:
    import app.dataService.globalVariable as GV
    import app.dataService.utils as utils
    import app.dataService.llm_eval as llmeval
    import app.dataService.preprocess as preprocess

ans_key = "answer_structure"

class DataService(object):
    def __init__(self):
        self.name = "dataService"
        self.GV = GV
        self.openai_key = GV.openai_key
        self.utils = utils

        self.paper_folder = GV.data_dir
        self.table_folder = GV.table_dir

        self.load_flag = True # load precomputed vectorstore and docstore
        self.vectorstore_loaded = False
        print("loading vectorstores...")
        self._load_vectorstores_(load_flag=self.load_flag)
        print("finished loading vectorstores")
        os.environ["OPENAI_API_KEY"] = GV.openai_key

    def _load_vectorstores_(self, load_flag=True):
        """
        Load or create vectorstores for each PDF file in the data directory.

        Args:
            load_flag (bool): If True, load pre-computed vectorstores. Otherwise, process PDFs and create new vectorstores.

        This method initializes retrievers for each PDF, which are used for efficient document retrieval.
        """
        if self.vectorstore_loaded:
            return
        data_folder = self.paper_folder
        table_folder = self.table_folder
        id_key = "doc_id"

        retrievers = {}
        for pdf_file in os.listdir(data_folder):
            if pdf_file.endswith(".pdf"):
                vectorstore_path = os.path.join(data_folder, "vectorstore", pdf_file.split(".")[0], "vector_index")
                db_path = os.path.join(data_folder, "vectorstore", pdf_file.split(".")[0], pdf_file.split(".")[0] + ".pickle")
                if load_flag: 
                    vectorstore = FAISS.load_local(vectorstore_path, 
                                                   embeddings=OpenAIEmbeddings(openai_api_key = GV.openai_key),
                                                   allow_dangerous_deserialization=True)
                    docstore = pickle.load(open(db_path, "rb"))
                else:
                    # if no precomputed vectorstores, process pdfs then
                    pdf_path = os.path.join(data_folder, pdf_file)
                    table_path = os.path.join(table_folder, pdf_file.split(".")[0] + ".json")
                    all_text = preprocess.process_one_pdf_papermage(pdf_path, table_path)
                    vectorstore, docstore = utils.build_local_document_vector_store(all_text, GV.openai_key)
                retriever = utils.build_multivector_retriever(vectorstore, docstore, id_key=id_key)
                retrievers[pdf_file] = retriever
        self.retrievers = retrievers

    def run_rag_qa(self, pdf_files: list, question: str, batch_size: int = 5, evaluation_metrics = None) -> tuple:
        """
        Run Retrieval Augmented Generation (RAG) for question answering on multiple PDF files.

        This method processes the given PDF files in parallel batches, retrieves relevant
        information, and generates answers using a language model.

        Args:
            pdf_files (list): List of PDF filenames to process.
            question (str): The question to be answered.
            batch_size (int, optional): Number of PDF files to process in parallel. Defaults to 5.
            evaluation_metrics (list, optional): Metrics to use for evaluating answers. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - rag_summary (str): A concise summary of answers from different papers.
                - results (dict): Detailed results for each PDF file, including answers and contexts.

        Raises:
            Exception: If there's an error processing any of the PDF files.
        """
        #######################
        # decide the answer strcuture
        #######################
        # decide the answer strcuture
        answer_structure_prompt = f"""
        Given the following question, design a structured data format to represent ONLY the information explicitly requested:

        Question: {question}

        Your task:
        1. Carefully analyze the question to identify ONLY the specific information explicitly requested or clearly implied. 
        2. Design a table structure with columns that directly correspond to the requested information.
        3. Provide the structure in a "record" format: [{{"column_1": "value_description", "column_2": "value_description", ...}}]
        4. Ensure all dictionary objects in the list share the same set of columns.
        5. Use clear and descriptive names for the columns.
        6. Avoid nested structures or hierarchical data - keep everything flat.

        IMPORTANT: Do NOT add any columns for information that is not specifically mentioned or clearly implied in the question. Stick ONLY to what is explicitly asked for.

        Guidelines:
        - Choose column names that are self-explanatory and follow a consistent naming convention.
        - Create columns ONLY for information directly mentioned or clearly implied in the question.
        - Do NOT add columns for information that might be related but is not specifically asked for.
        - Describe the expected values for each column (e.g., data type, format, range, units).
        - For columns with multiple possible values:
            * Create boolean columns for each option (e.g., "has_feature_X", "has_feature_Y").
            * Or use a numeric scale to indicate presence/absence or degree (e.g., 0-5 scale).
        - For columns with a limited set of possible values, list all possible options.
        - For numerical values (e.g., size, length, weight), specify the unit of measurement if relevant.
        - For date/time values, specify the expected format.

        Formulate your response as a JSON object with the key "{ans_key}", containing the designed structure.

        Example format:
        {{
        "{ans_key}": [
            {{
            "customer_id": "Integer: Unique identifier for the customer",
            "purchase_date": "String: Date of purchase in YYYY-MM-DD format",
            "total_amount": "Float: Total purchase amount in USD, rounded to 2 decimal places",
            "payment_method_credit_card": "Boolean: True if paid by credit card, False otherwise",
            "payment_method_debit_card": "Boolean: True if paid by debit card, False otherwise",
            "payment_method_paypal": "Boolean: True if paid by PayPal, False otherwise",
            "payment_method_cash": "Boolean: True if paid by cash, False otherwise",
            "is_member": "Boolean: True if the customer is a member, False otherwise",
            "product_category_electronics": "Boolean: True if electronics were purchased, False otherwise",
            "product_category_clothing": "Boolean: True if clothing was purchased, False otherwise",
            "product_category_groceries": "Boolean: True if groceries were purchased, False otherwise",
            "customer_satisfaction": "Integer: Customer satisfaction score on a scale of 1-5, where 1 is very unsatisfied and 5 is very satisfied"
            }}
        ]
        }}

        Ensure your structure capture all relevant information from the question, while also being flexible enough to accommodate various possible answers.
        """

        model = ChatOpenAI(temperature=0, 
                            model="gpt-4o-mini",
                            openai_api_key = GV.openai_key,
                            model_kwargs={
                                # "seed": 42,
                                "response_format": { "type": "json_object" }
                                }
                            )
        ans_format = json.dumps(json.loads(model.invoke(answer_structure_prompt).content))
        # print("ans_format: \n", ans_format)
        #######################


        retrievers = {}
        for pdf_file in pdf_files:
            retrievers[pdf_file] = self.retrievers[pdf_file]

        # Function to divide retrievers into batches for parallel processing
        def batched_retrievers(retriever_items, batch_size):
            for i in range(0, len(retriever_items), batch_size):
                yield retriever_items[i:i + batch_size]
        
        # Function to process each batch in parallel
        print("running rag retriever...")
        results = {}
        for batch in batched_retrievers(list(retrievers.items()), batch_size):
            time0 = time.time()
            with ThreadPoolExecutor() as executor:
                future_to_retriever = {executor.submit(self.process_rag_retriever, retriever_tuple, question, ans_format, evaluation_metrics): retriever_tuple for retriever_tuple in batch}
                for future in as_completed(future_to_retriever):
                    retriever_tuple = future_to_retriever[future]
                    # print(f"Processing {retriever_tuple[0]}, {retriever_tuple[1]}")
                    data = future.result()
                    # print(f"data: {data}")
                    results.update(data)
                    try:
                        data = future.result()
                        # print(f"data: {data}")
                        results.update(data)
                        # Process the data here
                    except Exception as exc:
                        print('%r generated an exception: %s' % (retriever_tuple[0], exc))
            time1 = time.time()
            print("Time taken for this batch: ", time1 - time0, " seconds")
            
        # Functions to summarize RAG results
        print("running rag summary...")
        summary_template = """
        Given the question: {question}, provide a very concise summary of the answers from different papers:
        {answer}
        """
        summary_prompt = ChatPromptTemplate.from_template(summary_template)
        summary_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106", openai_api_key = GV.openai_key)
        rag_summary_chain = summary_prompt | summary_model | StrOutputParser()
        rag_sum_text = utils.cut_string_to_token_length(str([str(r["answer"]) for r in list(results.values())]))
        
        rag_summary = rag_summary_chain.invoke({
            "question": question,
            "answer": rag_sum_text
        })
        return rag_summary, results

    
    def process_rag_retriever(self, retriever_item: tuple, 
                              question: str, answer_format: str, evaluation_metrics = None) -> dict:
        """
        Process a single retriever for RAG-based question answering.

        This method takes a retriever associated with a specific PDF file, retrieves relevant
        information, generates an answer, and optionally evaluates the answer quality.

        Args:
            retriever_item (tuple): A tuple containing (pdf_file: str, retriever: Retriever).
            question (str): The question to be answered.
            answer_format (str): The desired format for the answer, typically in JSON.
            evaluation_metrics (list, optional): List of metrics to evaluate the answer quality.
                Defaults to None.

        Returns:
            dict: A dictionary containing the results for the processed PDF file. Structure:
                {
                    pdf_file: {
                        "answer": dict,  # The structured answer
                        "context": {
                            "text": list,  # List of relevant text snippets
                            "tables": list,  # List of relevant tables
                            "figures": list  # List of relevant figures
                        },
                        "evaluation": dict  # Evaluation results (if metrics provided)
                    }
                }

        Raises:
            Exception: If there's an error in processing the retriever or generating the answer.

        Note:
            This method performs several steps:
            1. Retrieves relevant information using the provided retriever.
            2. Generates an answer using a language model.
            3. Classifies and processes the context (text, tables, figures).
            4. Optionally evaluates the answer quality.
        """
        pdf_file = retriever_item[0]
        retriever = retriever_item[1]
        chain = utils.build_rag_chain(retriever)

        response = chain.invoke({
                "question": question,
                "ans_format": answer_format
            },)
        answer = json.loads(response["answer"])[ans_key]
        context = response["context"]

        # LLM evaluation result according to the hyperparameter
        time1 = time.time()
        if evaluation_metrics != None:
            evaluation = llmeval.llm_evaluate_deepeval(metrics=evaluation_metrics, question=str(question), answer=str(answer), contexts=str(context))
        else:
            evaluation = None
        time2 = time.time()
        # print("****evaluation time****", time2-time1)

        # -----add two functions: judge the context type and retireve the table-----
        def check_type(context):
            # judge the context type
            table_pattern = r"(Table \d+).*"
            figure_pattern = r"(Figure \d+).*"

            table_match = re.search(table_pattern, context)
            figure_match = re.search(figure_pattern, context)

            if table_match:
                return {"value": "table", "content": table_match.group(1)}
            elif figure_match:
                return {"value": "figure", "content": figure_match.group(1)}
            else:
                return {"value": "text", "content": context}

        def find_matching_table(file_path, table_name):
            # retrieve the table from the table database
            with open(file_path, 'r') as file:
                data = json.load(file)
                matching_tables = {}
                for item in data:
                    if 'table_name' in item and item['table_name'] == table_name:
                        if 'table_caption' in item and 'table_content' in item:
                            matching_tables['table_caption'] = item['table_caption']
                            matching_tables['table_content'] = item['table_content']
                            return matching_tables
                matching_tables['table_caption'] = ""
                matching_tables['table_content'] = ""
            return matching_tables
        
        def find_matching_figure(file_path, figure_name):
            # retrieve the figure from the figure database
            with open(file_path, 'r') as file:
                data = json.load(file)
                matching_figures = {}
                for item in data:
                    if 'figure_name' in item and item['figure_name'] == figure_name:
                        if 'figure_caption' in item and 'figure_content' in item:
                            matching_figures['figure_caption'] = item['figure_caption']
                            matching_figures['figure_content'] = item['figure_content']
                            return matching_figures
                matching_figures['figure_caption'] = ""
                matching_figures['figure_content'] = ""
            return matching_figures
        
        def extract_table_and_text(context, table_pattern=r"(Table \d+)"):
            # Split the context into parts before, during, and after the table
            parts = re.split(table_pattern, context, maxsplit=1)
            non_table_text_before = parts[0].strip()
            table_text = parts[1].strip() if len(parts) > 1 else ""
            non_table_text_after = parts[2].strip() if len(parts) > 2 else ""
            return non_table_text_before, table_text, non_table_text_after
        # ----------------------------------end------------------------------------
        context_classification = {"text": [], "tables": [], "figures": []}
        added_figures = set()
        added_tables = set()    
        for c in context:
            type = check_type(c)
            if type["value"] == "text":
                context_classification["text"].append({
                    "content": type["content"]
                })
            elif type["value"] == "table":
                try:
                    retrieved_table = find_matching_table(os.path.join(GV.table_dir, retriever_item[0].split('.')[0] + '.json'), type["content"])
                    if retrieved_table and type["content"] not in added_tables:
                        # cover non-table string before and after the table string, and add them into the CONTEXTS
                        non_table_text, table_text, non_table_text_after = extract_table_and_text(c)
                        context_classification["text"].append({
                            "content": non_table_text + "\n ...\n" + non_table_text_after
                        })
                        # print("non_table_text", non_table_text, "\n ... \n", non_table_text_after)
                        context_classification["tables"].append({
                            "content": retrieved_table['table_content'],
                            "name": type["content"],
                            "caption": retrieved_table['table_caption']
                        })
                        added_tables.add(type["content"])
                except:
                    context_classification["text"].append({
                        "content": c
                    })
            elif type["value"] == "figure":
                try:
                    retrieved_figure = find_matching_figure(os.path.join(GV.figure_dir, retriever_item[0].split('.')[0] + '.json'), type["content"])
                    if retrieved_figure and type["content"] not in added_figures:
                        context_classification["figures"].append({
                            "content": retrieved_figure['figure_content'],
                            "name": type["content"],
                            "caption": retrieved_figure['figure_caption']
                        })
                        added_figures.add(type["content"])
                except:
                    context_classification["text"].append({
                        "content": c
                    })
        return { 
            pdf_file: {
                "answer": answer,
                "context": context_classification,
                "evaluation": evaluation
            }
        }
    
    


if __name__ == "__main__":
    print('=== dataService ===')
    dataService = DataService()
    #####################################
    # test run rag qa
    pdf_files =  [pf for pf in os.listdir(GV.data_dir) if pf.endswith(".pdf")]

    question = """
    {
        "model_name": model proposed in the paper,
        "model_size": how many parameters in the model,
        "pretrained_data_scale": pretrained data scale for the model,
        "hardware": hardware (GPU, TPU and how many) used for training the model,
        "evaluation": whether the model has been evaluated with in-context learning (ICL) and/or chain-of-thought (CoT)
    }
    """

    question = """
    What the tasks and accuracy of different LMs?
    """

    #******************** running **********************
    time1 = time.time()
    # rag_summary, ans = dataService.run_rag_qa(pdf_files, question, batch_size = 15, evaluation_metrics=['faithfulness', 'answer_relevancy', 'contextual_relevancy'])
    rag_summary, ans = dataService.run_rag_qa(pdf_files, question, batch_size = 15)
    time2 = time.time()
    print(time2-time1)
    print("ans: ", ans)
    print("\n")
    print("summary: ", rag_summary)

    
