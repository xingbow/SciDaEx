# SciDaEx Backend Routes
This folder contains the routing functions for the SciDaEx (Scientific Data Extractor) backend application.

## Files

- `api.py`: The main API routing file that defines the endpoints for the SciDaEx backend.

## Overview

The `api.py` file sets up a Flask Blueprint named `api` and defines various routes for handling different functionalities of the SciDaEx application. These routes include:

- File upload and retrieval
- PDF metadata extraction
- Table extraction from PDFs
- Figure extraction from PDFs
- Question answering (QA) functionality
- PDF categorization
- Document summarization
- Confidence score calculation

## Key Routes

- `/upload`: Handles file uploads
- `/extract_meta_from_pdf`: Extracts metadata from uploaded PDFs
- `/extract_table_from_pdf`: Extracts tables from PDFs
- `/extract_figure_from_pdf`: Extracts figures from PDFs
- `/qa`: Processes question-answering requests
- `/summarize`: Summarizes document content
- `/get_confidence_scores`: Calculates confidence scores for answers