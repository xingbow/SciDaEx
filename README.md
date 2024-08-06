# SciDaEx: Scientific Data Extraction and Structuring System

![SciDaEx Logo](scidaex_system.png)

SciDaEx is a open-source system for extracting and structuring data (as data tables) from scientific literature using Large Language Models (LLMs). It integrates a computational backend with an interactive user interface to facilitate efficient data extraction, structuring, and refinement for evidence synthesis in scientific research.

## Table of Contents

- [SciDaEx: Scientific Data Extraction and Structuring System](#scidaex-scientific-data-extraction-and-structuring-system)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
    - [Configuration](#configuration)
  - [Usage](#usage)
    - [Preprocess documents](#preprocess-documents)
    - [Running the web application](#running-the-web-application)
  - [Contact](#contact)

## Features

- Automated data extraction from scientific papers (text, tables, and figures)
- Structured data table output in standardized formats
- Interactive user interface for data validation and refinement
- Retrieval-augmented generation (RAG) for enhanced accuracy and speed
- Quality evaluation metrics for extracted data
- Support for both technical and non-technical users


## Installation

```bash
# Clone the repository
git clone https://github.com/xingbow/SciDaEx.git
cd SciDaEx

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install backend dependencies (python 3.10)
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Configuration
1. Backend configuration
   - Create a `config.yml` file in the `backend/app/dataService` directory
   - Update the `config.yml` file with the required configurations:
     - You can get adobe service api credentials [here](https://developer.adobe.com/document-services/docs/overview/pdf-services-api/)
     - You can get openai api key [here](https://platform.openai.com/api-keys)
    ```yaml
    api_keys:
       openai: your_openai_api_key

    adobe_credentials:
       client_id: your_adobe_client_id
       client_secret: your_adobe_client_secret
    ``` 

2. Frontend configuration
   - Update `yourAdobeClientId` with your Adobe client ID in `frontend/src/service/service.js`
     - You can get Adobe client ID [here](https://developer.adobe.com/document-services/docs/overview/pdf-embed-api/) (notice: you can set domain to localhost for testing)
     - **TODO**: replace Adobe PDF viewer with other open-source alternatives

## Usage

### Preprocess documents
1. Place your PDF documents in the `backend/app/dataService/data` directory.
2. Run the preprocessing script:
   ```bash
   cd backend/app/dataService
   python preprocess.py --pdf_dir data --table_dir data/table --figure_dir data/figure --meta_dir data/meta
   ```  
    This script will extract tables, figures, and metadata from the PDFs and store them in the respective directories.

For details, please refer to the [preprocessing documentation](backend/app/dataService/README.md).


### Running the web application
1. Start the backend server
   ```bash
   cd backend
   python run-data-backend.py
   ```

2. Start the frontend server
   ```bash
   cd frontend
   npm run serve
   ```
3. Open your browser and navigate to `http://localhost:8080` to access the SciDaEx interface.


## Contact

[Xingbo Wang](https://andy-xingbowang.com/) - xiw4011@med.cornell.edu




