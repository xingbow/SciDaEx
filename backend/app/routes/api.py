"""
API routes for the SciDaSynth backend service.

This module provides RESTful API endpoints for:
- File management (upload, download, listing)
- PDF processing (metadata, table, and figure extraction)
- Question answering and summarization
- Data analysis and clustering

The API follows RESTful conventions.
"""

# Standard library imports
import json
import os
import logging
import pickle
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from crypt import methods
import time
# Third-party library imports
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel, Field
from typing import List
import umap
# Flask imports
from flask import (
    Blueprint,
    current_app,
    request,
    jsonify,
    send_from_directory,
    url_for,
    Response,
    stream_with_context,
    abort
)

# LangChain imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.embeddings import OpenAIEmbeddings

# Local application imports
from app.dataService.llm_eval import llm_evaluate_deepeval
from app.dataService.utils import (
    extract_pdf_figure,
    extract_pdf_meta_information,
)
import app.dataService.summarize as summ
import base64

import openai
from openai import OpenAI

# Configure logging
LOG = logging.getLogger(__name__)


# Create Blueprint
api = Blueprint('api', __name__, url_prefix='/api')

# Pydantic models for request/response validation
class FileUploadResponse(BaseModel):
    url: str = Field(..., description="URL of the uploaded file")

class QAResponse(BaseModel):
    summary: str = Field(..., description="Summary of the answer")
    answer: Dict[str, Any] = Field(..., description="Detailed answer")

class ClusteringResponse(BaseModel):
    groupedDimensions: List[Dict[str, Any]] = Field(..., description="Clustered dimensions")
    combinedDimension: List[Dict[str, Any]] = Field(..., description="Combined dimension data")

# Error handlers
@api.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad request", "message": str(error)}), 400

@api.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found", "message": str(error)}), 404

@api.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "message": str(error)}), 500

# File management endpoints
@api.route('/files', methods=['GET'])
def get_files() -> Response:
    """
    Get a list of available PDF files.
    
    Returns:
        Response: JSON response containing list of files with their metadata
    """
    try:
        files = []
        data_dir = Path(current_app.dataService.GV.data_dir)
        
        for filename in data_dir.glob('*.pdf'):
            file_path = data_dir / filename
            with open(file_path, 'rb') as file:
                file_data = file.read()
                files.append({
                    "name": filename.name,
                    "url": request.host_url + f'api/uploads/{filename.name}',
                    "raw": "data:application/pdf;base64," + base64.b64encode(file_data).decode('utf-8')
                })
        return jsonify(files)
    except Exception as e:
        LOG.error(f"Error getting files: {str(e)}")
        abort(500, description=str(e))

@api.route('/upload', methods=['POST'])
def upload() -> Response:
    """
    Upload PDF files.
    
    Returns:
        Response: JSON response containing URLs of uploaded files
    """
    try:
        if 'file' not in request.files:
            abort(400, description="No file part in the request")
            
        files = request.files.getlist('file')
        file_urls = []
        
        for file in files:
            if file:
                filename = file.filename
                file_path = Path(current_app.dataService.GV.data_dir) / filename
                file.save(file_path)
                file_urls.append({
                    "url": request.host_url + f'api/uploads/{filename}'
                })
                
        return jsonify(file_urls)
    except Exception as e:
        LOG.error(f"Error uploading files: {str(e)}")
        abort(500, description=str(e))

@api.route('/uploads/<filename>')
def uploaded_file(filename: str) -> Response:
    """
    Serve uploaded files.
    
    Args:
        filename: Name of the file to serve
        
    Returns:
        Response: File content
    """
    try:
        return send_from_directory(current_app.dataService.GV.data_dir, filename)
    except Exception as e:
        LOG.error(f"Error serving file {filename}: {str(e)}")
        abort(404, description=f"File {filename} not found")

@api.route('/images/<filename>')
def serve_image(filename):
    # print("serve_image: ", current_app.dataService.GV.data_dir, "output",  (filename))
    return send_from_directory(current_app.dataService.GV.data_dir, "output/" +  (filename))

# PDF processing endpoints
@api.route("/extract_meta_from_pdf", methods=["POST"])
def extract_meta_from_pdf() -> Response:
    """
    Extract metadata from PDF files.
    
    Returns:
        Response: JSON response containing metadata for each file
    """
    try:
        data = request.get_json()
        if not data or "filenames" not in data:
            abort(400, description="No filenames provided")
            
        filenames = data["filenames"]
        meta_infos = []
        
        for filename in filenames:
            meta_path = Path(current_app.dataService.GV.meta_dir) / f"{filename['name'].split('.')[0]}.json"
            with open(meta_path, "r") as f:
                meta_infos.append(json.load(f))
                
        return jsonify(meta_infos)
    except Exception as e:
        LOG.error(f"Error extracting metadata: {str(e)}")
        abort(500, description=str(e))

@api.route("/extract_table_from_pdf", methods=["POST"])
def extract_table_from_pdf() -> Response:
    """
    Extract tables from PDF files.
    
    Returns:
        Response: JSON response containing tables for each file
    """
    try:
        data = request.get_json()
        if not data or "filenames" not in data:
            abort(400, description="No filenames provided")
            
        filenames = data["filenames"]
        table_infos = []
        
        for filename in filenames:
            table_path = Path(current_app.dataService.GV.table_dir) / f"{filename['name'].split('.')[0]}.json"
            with open(table_path, "r") as f:
                table_infos.append(json.load(f))
                
        return jsonify(table_infos)
    except Exception as e:
        LOG.error(f"Error extracting tables: {str(e)}")
        abort(500, description=str(e))

@api.route("/extract_figure_from_pdf", methods=["POST"])
def extract_figure_from_pdf() -> Response:
    """
    Extract figures from PDF files.
    
    Returns:
        Response: JSON response containing figures for each file
    """
    try:
        data = request.get_json()
        if not data or "filenames" not in data:
            abort(400, description="No filenames provided")
            
        filenames = data["filenames"]
        filepaths = [Path(current_app.dataService.GV.data_dir) / filename["name"] for filename in filenames]
        extract_figures = [extract_pdf_figure(str(filepath)) for filepath in filepaths]

        for figs in extract_figures:
            for curr_fig in figs:
                figure_name = curr_fig["figure_url"].split("/")[-1]
                fig_url = url_for('api.serve_image', filename=figure_name, _external=True)
                curr_fig["figure_url"] = fig_url
                
        return jsonify(extract_figures)
    except Exception as e:
        LOG.error(f"Error extracting figures: {str(e)}")
        abort(500, description=str(e))

# Question answering and analysis endpoints
@api.route('/qa', methods=["POST"])
def qa() -> Response:
    """
    Perform question answering on PDF files.
    
    Returns:
        Response: JSON response containing summary and answer
    """
    try:
        data = request.json
        if not data or "question" not in data or "filenames" not in data:
            abort(400, description="Missing required fields: question or filenames")
            
        question = data["question"]
        filenames = [filename["name"] for filename in data["filenames"]]
        summary, ans = current_app.dataService.run_rag_qa(filenames, question, batch_size=75)

        return jsonify({
            "summary": summary,
            "answer": ans
        })
    except Exception as e:
        LOG.error(f"Error in question answering: {str(e)}")
        abort(500, description=str(e))

@api.route('/summarize', methods=["POST"])
def summarize() -> Response:
    """
    Summarize documents.
    
    Returns:
        Response: JSON response containing summary
    """
    try:
        data = request.json
        if not data:
            abort(400, description="No data provided")
            
        docs = [
            Document(
                page_content=f"paper title: {d['title']}; paper abstract: {d['abstract']}",
                metadata={"source": d['title']},
            )
            for d in data
        ]
        return jsonify(summ.summarize_docs(docs))
    except Exception as e:
        LOG.error(f"Error in summarization: {str(e)}")
        abort(500, description=str(e))

@api.route('/get_confidence_scores', methods=['POST'])
def get_eval_scores() -> Response:
    """
    Get confidence scores for answers.
    
    Returns:
        Response: JSON response containing confidence scores
    """
    try:
        data = request.json
        if not data or "question" not in data or "answer" not in data:
            abort(400, description="Missing required fields: question or answer")
            
        question = data['question']
        answer = str(data['answer'])
        return jsonify(llm_evaluate_deepeval(metric=['answer_relevancy'], question=question, answer=answer, contexts=""))
    except Exception as e:
        LOG.error(f"Error getting confidence scores: {str(e)}")
        abort(500, description=str(e))

# Clustering and analysis functions
def perform_clustering(values: List[Any], column_name: str) -> Dict[str, Any]:
    """
    Perform clustering on values.
    
    Args:
        values: List of values to cluster
        column_name: Name of the column
        
    Returns:
        Dict containing clustered data
    """
    if all(isinstance(v, (int, float)) for v in values):
        return cluster_numerical_values(values, column_name)
    else:
        return cluster_qualitative_values(values, column_name)

def cluster_numerical_values(values: List[Union[int, float]], column_name: str) -> Dict[str, Any]:
    """
    Cluster numerical values using quantiles.
    
    Args:
        values: List of numerical values
        column_name: Name of the column
        
    Returns:
        Dict containing clustered data
    """
    quantiles = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    
    data = []
    for i, label in enumerate(labels):
        if i == 0:
            cluster_values = [v for v in values if v <= quantiles[0]]
        elif i == 4:
            cluster_values = [v for v in values if v > quantiles[3]]
        else:
            cluster_values = [v for v in values if quantiles[i-1] < v <= quantiles[i]]

        for value in cluster_values:
            data.append({
                "value": str(value),
                "count": 1,
                "position": [i, value],
                "cluster": label
            })
    
    return {
        "name": column_name,
        "data": data
    }

def cluster_qualitative_values(values: List[str], column_name: str) -> Dict[str, Any]:
    """
    Cluster qualitative values using embeddings and UMAP.
    
    Args:
        values: List of string values
        column_name: Name of the column
        
    Returns:
        Dict containing clustered data
    """
    unique_values = list(set(map(str, values)))
    embeddings = get_embeddings(unique_values)
    embeddings_array = np.array(embeddings)

    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings_array)

    n_clusters = min(5, len(unique_values))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(umap_embeddings)

    unique_labels = set(labels)
    clustered_values = [[] for _ in range(len(unique_labels))]
    for i, label in enumerate(labels):
        clustered_values[label].append(unique_values[i])

    cluster_labels = generate_cluster_labels(clustered_values, column_name)
    value_counts = {value: values.count(value) for value in unique_values}

    final_data = []
    for i, (value, position) in enumerate(zip(unique_values, umap_embeddings)):
        cluster_index = labels[i]
        final_data.append({
            "value": value,
            "count": value_counts[value],
            "position": position.tolist(),
            "cluster": cluster_labels[cluster_index] if cluster_index < len(cluster_labels) else f"Group {cluster_index + 1}"
        })

    return {
        "name": column_name,
        "data": final_data
    }

def get_embeddings(values: List[str]) -> List[List[float]]:
    """
    Get embeddings for a list of values using OpenAI's API.
    
    Args:
        values: List of strings to embed
        
    Returns:
        List of embeddings
    """
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=values
    )
    return [item.embedding for item in response.data]

def generate_cluster_labels(clusters: List[List[str]], column_name: str) -> List[str]:
    """
    Generate descriptive labels for clusters using OpenAI's API.
    
    Args:
        clusters: List of clusters, where each cluster is a list of values
        column_name: Name of the column
        
    Returns:
        List of cluster labels
    """
    prompt = f"""Given the following clusters of values from the '{column_name}' columns, please provide a short, descriptive label for each cluster describe columns' values characteristics. Return the result as a JSON object with the following structure:
    {{
        "labels": [
            "Label for Cluster 1",
            "Label for Cluster 2",
            ...
        ]
    }}
    Ensure that each label is concise and accurately represents the values in its cluster.
    The number of labels should match the number of clusters provided.

    Clusters:
    {json.dumps(clusters, indent=2)}
    """

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert at analyzing and labeling data clusters."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.2
    )

    result = json.loads(response.choices[0].message.content)
    return result.get('labels', [])

@api.route('/prepare_grouped_data', methods=['POST'])
def prepare_grouped_data() -> Response:
    """
    Prepare grouped data for visualization.
    
    Returns:
        Response: JSON response containing grouped dimensions and combined dimension data
    """
    try:
        data = request.json
        if not data or "qa_table_data" not in data:
            abort(400, description="Missing required field: qa_table_data")
            
        qa_table_data = data['qa_table_data']
        selected_columns = data.get('selected_columns', [])

        dimensions = []

        if qa_table_data:
            for column in selected_columns:
                if column in qa_table_data[0]:
                    values = [row[column] for row in qa_table_data]
                    clustered_data = perform_clustering(values, column)
                    dimensions.append(clustered_data)

            if len(selected_columns) == 1:
                return jsonify({"groupedDimensions": dimensions, "combinedDimension": []})

            if len(selected_columns) > 1:
                df = pd.DataFrame(qa_table_data)
                combined_embedding, combined_labels = generate_combined_embedding(df[selected_columns])
                combined_cluster_labels = generate_cluster_labels(
                    [df[selected_columns][combined_labels == i].values.tolist() for i in range(len(set(combined_labels)))],
                    "|".join(selected_columns)
                )
                
                combined_dimension = [{
                    "name": "Combined Columns",
                    "data": [
                        {
                            "value": ", ".join(str(df[col].iloc[i]) for col in selected_columns),
                            "count": 1,
                            "position": combined_embedding[i].tolist(),
                            "cluster": combined_cluster_labels[label] if label < len(combined_cluster_labels) else f"Group {label + 1}"
                        }
                        for i, label in enumerate(combined_labels)
                    ]
                }]

                return jsonify({"groupedDimensions": dimensions, "combinedDimension": combined_dimension})

        return jsonify({"groupedDimensions": dimensions, "combinedDimension": []})
    except Exception as e:
        LOG.error(f"Error preparing grouped data: {str(e)}")
        abort(500, description=str(e))

def generate_combined_embedding(df: pd.DataFrame) -> tuple:
    """
    Generate combined embeddings for multiple columns.
    
    Args:
        df: DataFrame containing the columns to combine
        
    Returns:
        Tuple of (embeddings, labels)
    """
    combined_values = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            quantiles = np.quantile(df[column], [0.2, 0.4, 0.6, 0.8])
            labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
            column_values = pd.cut(df[column], bins=[-np.inf] + list(quantiles) + [np.inf], labels=labels)
        else:
            column_values = df[column]
        combined_values.append(column_values.astype(str))
    
    combined_strings = [' | '.join(row) for row in zip(*combined_values)]
    embeddings = get_embeddings(combined_strings)
    embeddings_array = np.array(embeddings)

    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings_array)

    n_clusters = min(5, len(combined_strings))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings_array)

    return umap_embeddings, labels

if __name__ == '__main__':
    pass
