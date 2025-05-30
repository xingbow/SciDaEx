# SciDaEx Backend

This is the backend component of the SciDaEx (Scientific Data Extractor) project. It provides a Flask-based API for processing scientific papers, extracting metadata, tables, and figures, and performing various analysis tasks.

## Project Structure

```
SciDaEx/backend/
├── app/
│   ├── dataService/
│   │   ├── dataService.py
│   │   ├── globalVariable.py
│   │   ├── llm_eval.py
│   │   ├── preprocess.py
│   │   ├── summarize.py
│   │   └── utils.py
│   ├── routes/
│   │   └── api.py
│   └── app.py
├── README.md
└── run-data-backend.py
```

## Key Components

- `app.py`: The main Flask application setup.
- `routes/api.py`: Defines the API endpoints.
- `dataService/`: Contains the core functionality for data processing and analysis.

