import os
import pickle
import json
import time
import re
try:
    import globalVariable as GV
    import utils as utils
except:
    import app.dataService.globalVariable as GV
    import app.dataService.utils as utils

from unstructured.partition.pdf import partition_pdf
from tqdm import tqdm 
import argparse

def split_into_chunks(text, 
                      section_boundaries, 
                      excluded_ranges, 
                      max_chunk_size = 4000, 
                      target_chars = 3800, 
                      combine_text_under_n_chars = 2000):
    # Function to check if a given index falls within any excluded ranges
    def is_excluded(index):
        for start, end in excluded_ranges:
            if start <= index < end:
                return True
        return False

    # Function to merge small chunks with their neighbors
    def merge_small_chunks(chunks):
        merged_chunks = []
        buffer = ""
    
        for chunk in chunks:
            if len(buffer) + len(chunk) < combine_text_under_n_chars:
                buffer += chunk
            else:
                if len(buffer) < combine_text_under_n_chars:  # If buffer is small, merge it with the current chunk
                    buffer += chunk
                else:  # Otherwise, add the buffer as a separate chunk
                    merged_chunks.append(buffer)
                    buffer = chunk
    
        if buffer:  # Add any remaining buffer
            merged_chunks.append(buffer)
    
        return merged_chunks

    # Split the text into sections while excluding specified ranges
    sections = []
    for start, end in section_boundaries:
        section_text = ""
        for i in range(start, end):
            if not is_excluded(i):
                section_text += text[i]
        sections.append(section_text)

    # Split each section into chunks of approximately target_chars characters
    chunks = []
    for section in sections:
        for i in range(0, len(section), target_chars):
            chunk = section[i:i + max_chunk_size]  # Ensure chunk is no more than max_chunk_size chars
            if len(chunk) > combine_text_under_n_chars or (len(chunk) > 0 and i + max_chunk_size >= len(section)):
                chunks.append(chunk)

    # Merge chunks smaller than 2000 characters
    chunks = merge_small_chunks(chunks)

    return chunks

def process_one_pdf_papermage(pdf_path, table_path, figure_path, flag='all'):
    from papermage.recipes import CoreRecipe
    recipe = CoreRecipe()
    doc = recipe.run(pdf_path)
    # process the doc text
    # --- get section boundaries
    section_boundaries = []

    section_idxs = [0] + [sec.start for sec in doc.sections] + [len(doc.symbols)]
    for sec_num, sec_start in enumerate(section_idxs):
        if sec_num + 1 <len(section_idxs):
            section_boundaries.append((sec_start, section_idxs[sec_num+1]))
    # --- exclude references
    excluded_ranges = []
    for ref in doc.bibliographies:
        excluded_ranges.append((ref.start, ref.end))
    # --- split into chunks
    chunks = split_into_chunks(doc.symbols, section_boundaries, excluded_ranges)

    all_text = chunks

    if flag in ['all', 'table']:
        # --- load the table
        table_data = json.load(open(table_path))
        tables = []
        for table in table_data:
            table_text = f"""{table["table_name"]}: {table["table_caption"]}; table content: {table["table_content"]}
            """
            tables.append(table_text)
        all_text += tables
    
    if flag in ['all', 'figure']:
        # --- load the figure
        figure_data = json.load(open(figure_path))
        figures = []
        for figure in figure_data:
            figure_text = f"""{figure["figure_name"]}: {figure["figure_caption"]}; figure content: {figure["figure_content"]}
            """
            figures.append(figure_text)
        all_text += figures
    
    return all_text

def clean_text(text):
    # Use regex to remove (cid:xxx) patterns
    cleaned_text = re.sub(r'\(cid:\d+\)', '', text)
    return cleaned_text

def split_text(text, max_chunk_size):
    return [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

def process_text_chunks(text_list, max_chunk_size=4000, target_chars=3800, combine_text_under_n_chars=2000):
    processed_chunks = []
    temp_chunk = ""

    for text in text_list:
        # Split text if it exceeds max_chunk_size
        if len(text) > max_chunk_size:
            split_chunks = split_text(text, max_chunk_size)
            for chunk in split_chunks:
                if len(temp_chunk) + len(chunk) <= target_chars:
                    temp_chunk += chunk
                else:
                    if temp_chunk:
                        processed_chunks.append(temp_chunk)
                    temp_chunk = chunk
            if temp_chunk:
                processed_chunks.append(temp_chunk)
                temp_chunk = ""
        else:
            if len(temp_chunk) + len(text) <= target_chars:
                temp_chunk += text
            else:
                if temp_chunk:
                    processed_chunks.append(temp_chunk)
                temp_chunk = text

    # Add any remaining temp_chunk to the processed_chunks list
    if temp_chunk:
        processed_chunks.append(temp_chunk)

    # Second pass: Combine smaller chunks if they are under combine_text_under_n_chars
    combined_chunks = []
    temp_chunk = ""

    for chunk in processed_chunks:
        if len(chunk) < combine_text_under_n_chars:
            if len(temp_chunk) + len(chunk) <= target_chars:
                temp_chunk += chunk
            else:
                if temp_chunk:
                    combined_chunks.append(temp_chunk)
                temp_chunk = chunk
        else:
            if temp_chunk:
                combined_chunks.append(temp_chunk)
                temp_chunk = ""
            combined_chunks.append(chunk)

    # Add any remaining temp_chunk to the combined_chunks list
    if temp_chunk:
        combined_chunks.append(temp_chunk)

    return combined_chunks

def process_one_pdf(pdf_path, table_path, figure_path, flag='all'):
    """
    Process one pdf file and save the results to the table folder
    Input:
        pdf_path: path to the pdf file
        table_path: path to the table file
        figure_path: path to the figure file
    Output:
        None
    """
    # partition pdf text content
    # Get elements
    raw_pdf_elements = partition_pdf(
        filename= pdf_path,
        # Using pdf format to find embedded image blocks
        extract_images_in_pdf=False,
        # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
        # Titles are any sub-section of the document
        infer_table_structure=False,
        # Post processing to aggregate text once we have the title
        chunking_strategy="by_title",
        # Chunking params to aggregate text blocks
        # Attempt to create a new chunk 3800 chars
        # Attempt to keep chunks > 2000 chars
        # Hard max on chunks
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000
    )
    # Apply to text (postdoc processing to remove some unwanted characters and split the text into chunks)
    texts = []
    for ele in raw_pdf_elements:
        text = clean_text(str(ele))
        texts.append(text)
    # Split text into chunks
    all_text = process_text_chunks(texts)

    if flag in ['all', 'table']:
        # Load table
        table_data = json.load(open(table_path))
        tables = []
        for table in table_data:
            table_text = f"""{table["table_name"]}: {table["table_caption"]}; table content: {table["table_content"]}
            """
            tables.append(table_text)
        all_text += tables
    
    if flag in ['all', 'figure']:
        # Load the figure
        figure_data = json.load(open(figure_path))
        figures = []
        for figure in figure_data:
            figure_text = f"""{figure["figure_name"]}: {figure["figure_caption"]}; figure content: {figure["figure_content"]}
            """
            figures.append(figure_text)
        all_text += figures
    print("all text: ", all_text)
    print("*"*20)
    return all_text

def preprocess_folder(pdf_dir, figure_dir, table_dir, meta_dir, table_model, figure_model, meta_model, mode, openai_key, vectorstore_dir, flag):
    data_folder = pdf_dir
    table_folder = table_dir
    figure_folder = figure_dir
    vectorstore_dir = vectorstore_dir
    meta_folder = meta_dir
    mode = mode
    
    failed_files = []
    if mode == "fast":
        table_model = "none"
        figure_model = "none"

    if flag in ["all", "figure"]:
        # process figures
        utils.process_figures(data_folder, figure_folder, figure_model, openai_key)

    if flag in ["all", "table"]:
        # process tables
        utils.process_tables(data_folder, table_folder, table_model, openai_key)
    
    # utils.process_meta_information(data_folder, meta_folder, meta_model, openai_key)

    # start to generate vector stores
    for filename in tqdm(os.listdir(data_folder)):
        if not filename.endswith(".pdf"):
            continue
        try:
            pdf_path = os.path.join(data_folder, filename)
            table_path = os.path.join(table_folder, filename.split(".")[0] + ".json")
            figure_path = os.path.join(figure_folder, filename.split(".")[0] + ".json")
            vectorstore_path = os.path.join(vectorstore_dir, filename.split(".")[0], "vector_index")
            db_path = os.path.join(vectorstore_dir, filename.split(".")[0], filename.split(".")[0] + ".pickle")
            if not os.path.exists(vectorstore_path) or not os.path.exists(db_path):
                all_text = process_one_pdf(pdf_path, table_path, figure_path, flag)
                print(f"Processing {filename}")
                utils.save_local_document_vector_store(all_text, vectorstore_path, db_path, openai_key)
        except Exception as e:
            print(f"Failed to process {filename}")
            print(e)
            failed_files.append(filename)
    print(f"Failed files: {failed_files}")
    print("Preprocessing done.")

def preprocess_single_pdf(pdf_path, figure_dir, table_dir, meta_dir, table_model, figure_model, meta_model, mode, openai_key, vectorstore_dir, flag):
    pdf_path = pdf_path
    table_folder = table_dir
    figure_folder = figure_dir
    meta_folder = meta_dir
    vectorstore_dir = vectorstore_dir
    mode = mode
    
    if not os.path.exists(pdf_path):
        print(f"File {pdf_path} does not exist.")
        return
    elif not pdf_path.endswith(".pdf"):
        print(f"File {pdf_path} is not a pdf file.")
        return
    
    if mode == "fast":
        # if mode parameter is "fast", then set table_model to "none" to use the adobe function
        table_model = "none"
        figure_model = "none"
    
    if flag in ["all", "figure"]:
        # process figures
        utils.process_single_pdf_figure(pdf_path, figure_folder, figure_model, openai_key)

    if flag in ["all", "table"]:
        # process tables
        utils.process_single_pdf_table(pdf_path, table_folder, table_model, openai_key)
    
    # process meta information
    utils.process_single_pdf_meta_information(pdf_path, meta_folder, meta_model, openai_key)

    # start to generate vector stores
    try:
        table_path = os.path.join(table_dir, os.path.basename(pdf_path).split(".")[0] + ".json")
        figure_path = os.path.join(figure_dir, os.path.basename(pdf_path).split(".")[0] + ".json")
        vectorstore_path = os.path.join(vectorstore_dir, os.path.basename(pdf_path).split(".")[0], "vector_index")
        db_path = os.path.join(vectorstore_dir, os.path.basename(pdf_path).split(".")[0], os.path.basename(pdf_path).split(".")[0] + ".pickle")
        if not os.path.exists(vectorstore_path) or not os.path.exists(db_path):
            all_text = process_one_pdf_papermage(pdf_path, table_path, figure_path, flag)
            print(f"Processing {os.path.basename(pdf_path)}")
            utils.save_local_document_vector_store(all_text, vectorstore_path, db_path, openai_key)
    except Exception as e:
        print(f"Failed to process {pdf_path}")
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PDF files.")
    parser.add_argument('--pdf_dir', type=str, required=False, help='Directory containing PDF files.', default=GV.data_dir)
    parser.add_argument('--figure_dir', type=str, required=False, help='Directory to save extracted figures.', default=GV.figure_dir)
    parser.add_argument('--table_dir', type=str, required=False, help='Directory to save extracted tables.', default=GV.table_dir)
    parser.add_argument('--meta_dir', type=str, required=False, help='Directory to save extracted meta information.', default=GV.meta_dir)
    parser.add_argument('--table_model', type=str, required=False, help='Model to use for table extraction.', default="gpt-4o")
    parser.add_argument('--figure_model', type=str, required=False, help='Model to use for figure extraction.', default="gpt-4o-mini")
    parser.add_argument('--meta_model', type=str, required=False, help='Model to use for meta information extraction.', default="gpt-3.5-turbo-1106")
    parser.add_argument('--fast', action='store_true', default=False, help='Use fast mode (LLM) to extract tables')
    parser.add_argument('--openai_key', type=str, required=False, help='OpenAI API key.', default=GV.openai_key)
    parser.add_argument('--vectorstore_dir', type=str, required=False, help='Directory for vector store.', default=GV.vectorstore_dir)
    parser.add_argument('--pdf_path', type=str, required=False, help='Path to a single PDF file to process.')
    parser.add_argument('--flag', type=str, choices=['all', 'table', 'figure', "none"], default='all', help='Specify which elements to process: all, table, figure, none (using text only)')

    args = parser.parse_args()
    if args.fast:
        mode = "fast"
    else:
        mode = "normal"

    # print(mode)
    if args.pdf_path:
        preprocess_single_pdf(
            pdf_path=args.pdf_path,
            figure_dir=args.figure_dir,
            table_dir=args.table_dir,
            meta_dir=args.meta_dir,
            table_model=args.table_model,
            figure_model=args.figure_model,
            meta_model=args.meta_model,
            mode=mode,
            openai_key=args.openai_key,
            vectorstore_dir=args.vectorstore_dir,
            flag=args.flag
        )
    else:
        preprocess_folder(
            pdf_dir=args.pdf_dir,
            figure_dir=args.figure_dir,
            table_dir=args.table_dir,
            table_model=args.table_model,
            meta_dir=args.meta_dir,
            figure_model=args.figure_model,
            meta_model=args.meta_model,
            mode=mode,
            openai_key=args.openai_key,
            vectorstore_dir=args.vectorstore_dir,
            flag=args.flag
        )
