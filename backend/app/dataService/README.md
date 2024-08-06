# SciDaEx Data Service

This folder contains the core data processing and extraction functionalities for the SciDaEx (Scientific Data Extraction) project. These modules work together to process scientific papers, extract relevant information, and provide a question-answering capability.

## Key Features

- PDF processing and information extraction (`preprocess.py`)
  - Table and figure extraction from scientific papers
  - Meta-information extraction from papers
- Vector store creation and management (`dataService.py`)
- RAG-based question-answering system (`dataService.py`)
- LLM-based summarization (`summarize.py`)
- Evaluation metrics for QA performance (`llm_eval.py`)
- Global configuration and prompt management (`globalVariable.py`)
- Utility functions for various processing tasks (`utils.py`)

## Setup

1. Ensure all required libraries are installed (see requirements.txt in the parent directory).
2. Create a `config.yml` file in this directory with the following structure:
   ```yaml
   api_keys:
     openai: your_openai_key_here
     langchain: your_langchain_key_here

   adobe_credentials:
     client_id: your_adobe_client_id_here
     client_secret: your_adobe_client_secret_here
    ```
   - Replace the placeholder values in `config.yml` with your actual API keys and credentials.

## Usage

### Preprocess PDFs

You can use `preprocess.py` to process either a folder of PDFs or a single PDF file: 

1. Processing a folder of PDFs:
   ```python
   python preprocess.py \
   --pdf_dir <path_to_pdf_folder> \
   --figure_dir <path_to_figure_output_folder> \
   --table_dir <path_to_table_output_folder> \
   --meta_dir <path_to_meta_output_folder> \
   --openai_key <your_openai_api_key> \
   --vectorstore_dir <path_to_vectorstore_output_folder>
   ```

2. Processing a folder of PDFs:
   ```python
   python preprocess.py \
   --pdf_path <path_to_single_pdf_file> \
   --figure_dir <path_to_figure_output_folder> \
   --table_dir <path_to_table_output_folder> \
   --meta_dir <path_to_meta_output_folder> \
   --openai_key <your_openai_api_key> \
   --vectorstore_dir <path_to_vectorstore_output_folder>
   ```
Add the `--fast` flag for faster, non-LLM-based table extraction. For more options, run python preprocess.py --help.

### Using dataService.py

The `DataService` class in `dataService.py` provides the main question-answering functionality:

1. Ensure you have preprocessed your PDF files using `preprocess.py` as described in the previous section.

2. To use the DataService, you can refer to the example in the `dataService.py` file or use the following template:
   ```python
   from dataService import DataService

   # Initialize the DataService
   data_service = DataService()

   # Specify the PDF file names you want to query (NOTE: These files should have been preprocessed)
   pdf_files = ["example1.pdf", "example2.pdf", ...]

   # Your question
   question = "Your question here"

   # Run the QA system
   summary, results = data_service.run_rag_qa(pdf_files, question)

   # Process and use the results as needed
   print(summary)
   for pdf, result in results.items():
       print(f"Results for {pdf}:", result)
   ```
