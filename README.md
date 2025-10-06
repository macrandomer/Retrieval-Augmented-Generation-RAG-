# Retrieval-Augmented Generation (RAG) GPU Project

**Author:** Mohammed Abdul Aziz  
**Contact:** <mohdabdul532@gmail.com>

---

## Overview

This project implements an advanced Retrieval-Augmented Generation (RAG) system combining semantic search and natural language generation optimized with ONNX and GPU acceleration. It improves response accuracy for knowledge-based queries by retrieving relevant documents and generating coherent, grounded answers.

The system includes:  

- Vector-based semantic search with SentenceTransformer & FAISS  
- ONNX-exported DistilBART model optimized for GPU inference  
- Interactive CLI and Streamlit web UI interfaces  
- Advanced prompt engineering with few-shot examples and fallback handling  

---

## Features

- **Semantic Retrieval:** Fast and accurate retrieval of relevant documents using dense embeddings.  
- **Optimized Generation:** ONNX Runtime with GPU support enables efficient transformer inference.  
- **Interactive Web UI:** Streamlit app for real-time querying with display of retrieval and generation results.  
- **Robust Prompting:** Few-shot examples and fallback responses improve answer quality and user experience.  
- **Flexible Deployment:** Ready for local testing, containerization, and scalable cloud deployment.

---

## Installation

1. Clone the repository:

git clone <your-repo-url>
cd rag_project

text

2. Create and activate conda environment:

conda create -n rag_gpu_env python=3.10 -y
conda activate rag_gpu_env

text

3. Install dependencies:

pip install -r requirements.txt

text

---

## Project Structure

rag_project/
│
├── app.py # Streamlit web UI with advanced prompt engineering
├── vector_generation_module.py # Core vector search and ONNX generation logic
├── rag_gpu_project.py # CLI interactive RAG pipeline with improved fallbacks
├── export_onnx_model.py # Script to export DistilBART model to ONNX
├── distilbart.onnx # ONNX model file
├── requirements.txt # Project dependencies
├── README.md # This documentation

text

---

## Running the CLI Application

python rag_gpu_project.py

text

- Interactively enter questions.
- System retrieves documents and generates answers with smart fallback.
- Suitable for quick testing and terminal demos.

---

## Running the Web UI

streamlit run app.py

text

- Opens browser UI at `http://localhost:8501`.
- Enter queries into text box and view retrieval & generated answers.
- Useful for stakeholder demos and user-friendly interfaces.

---

## Prompt Engineering and Fallbacks

The system uses carefully designed prompts with few-shot examples of question-context-answer triples to instruct the model. It also features:

- Confidence thresholding on retrieval scores to avoid irrelevant generation.
- Polite fallback responses when questions are out of scope.
- Post-processing heuristics to clean and truncate verbose outputs.

---

## Deployment Recommendations

- Build a Docker container with all dependencies and Streamlit app.
- Deploy on GPU-enabled cloud instances (AWS, Azure, GCP) for best performance.
- Use Kubernetes for scalable production deployments with GPU autoscaling.
- Implement caching layers for embedding vectors and query results to reduce inference time.

---

## Sample Queries

| Query                                  | Expected Answer                                              |
|---------------------------------------|--------------------------------------------------------------|
| What safety gear must employees wear? | Employees must wear helmets and PPE like gloves and goggles. |
| What are the visitor requirements?    | Visitors must sign in at reception and wear visitor badges.  |
| Who can access the fire exits?         | Fire exits must be kept clear at all times for safe exit.    |

---

## Architecture & Design

### System Overview

![System Architecture](diagrams/system_architecture.png)

### Data Processing Pipeline

![Data Flow](diagrams/data_flow_diagram.png)

### Model Pipeline

![ML Pipeline](diagrams/model_pipeline.png)

### Deployment Architecture

![Deployment](diagrams/deployment_architecture.png)

For more detailed diagrams, see the [diagrams folder](diagrams/).

## Contact

For questions, feature requests, or collaboration opportunities, please contact:

**Mohammed Abdul Aziz**  
Email: <mohdabdul532@gmail.com>

---

## License

This project is licensed under the MIT License.

---

*Thank you for exploring this RAG GPU Project. Your interest and feedback are appreciated!*
