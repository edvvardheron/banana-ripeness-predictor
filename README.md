Banana Ripeness Predictor: End-to-End MLOps Pipeline

A production-ready computer vision system that predicts the exact ripeness (in days) of a banana from an image.

This repository serves as a reference implementation of a complete Machine Learning Operations (MLOps) lifecycle. It moves beyond standard model training to demonstrate data versioning, containerized inference, CI/CD automation, and continuous data drift monitoring.

Live Web App Demo (https://edvvardheron.github.io/banana-ripeness-predictor/frontend/)

System Architecture...

The project utilizes a hybrid cloud architecture, separating the static frontend from the containerized inference backend:

Frontend: A responsive, drag-and-drop HTML/JS interface hosted on GitHub Pages.

Inference API: A FastAPI web server wrapped in a Docker container, deployed serverlessly on Google Cloud Run.

Model: A custom regression model built via Transfer Learning on MobileNetV2 (TensorFlow/Keras).

Data Pipeline: Orchestrated locally via DVC (Data Version Control) with artifacts stored securely in Google Cloud Storage (GCS).

Key Production Features...

1. Erroneous Data Prevention

To ensure only clean data can be used to update the model, and ensure user security, images received are not stored within a Google Cloud Storage bucket. Only images continuously uploaded by myself are added for future model retraining.

2. Embedding Drift Detection

The system includes an automated monitoring script (src/detect_drift.py) to measure distribution shifts between the original training data and live production data. It extracts the high-level feature embeddings from the model's pooling layer and calculates the Wasserstein Distance to alert engineers to semantic drift before it degrades model performance.

3. CI/CD Automation

Continuous Integration (ci.yaml): On every push, GitHub Actions provisions a runner, installs dependencies, and executes pytest unit tests (verifying API health and tensor preprocessing shapes) and flake8 linting.

Continuous Deployment (cd.yaml): Upon merging to main, GitHub Actions automatically builds the Docker image and pushes the latest version to the GitHub Container Registry (GHCR), readying it for Cloud Run deployment.

Tech Stack...

Machine Learning: TensorFlow, Keras, SciPy, Pillow, NumPy

MLOps Tooling: DVC (Data Version Control), MLflow (Experiment Tracking)

Backend & Serving: FastAPI, Uvicorn, Docker

Cloud & CI/CD: Google Cloud Platform (Cloud Run, GCS), GitHub Actions

Local Development Setup...

Prerequisites

Docker

Python 3.10+

Google Cloud CLI (gcloud) authenticated to your project

1. Clone & Install

git clone https://github.com/edvvardheron/banana-ripeness-predictor.git
cd banana-ripeness-predictor
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt


2. Pull Data & Reproduce Pipeline

The raw image data and trained model artifacts are versioned via DVC.

# Pull the data from Google Cloud Storage
dvc pull

# Execute the DAG pipeline (prepare_data.py -> train_transfer.py)
dvc repro


3. Run the Inference API Locally

python src/app.py


Navigate to http://127.0.0.1:8000/docs to test the endpoints via the Swagger UI.

4. Run the Drift Detector

To test the monitoring system against incoming data:

# Add sample images to data/incoming/ then run:
python src/detect_drift.py


Run via Docker...

You can run the fully containerized API without setting up a Python environment:

docker build -t banana-predictor .
docker run -p 8000:8000 banana-predictor


Model Performance...

Architecture: MobileNetV2 (Base, Frozen) -> GlobalAveragePooling2D -> Dense(128, ReLU) -> Dropout(0.2) -> Dense(1, Linear).

Evaluation Metric: Mean Absolute Error (MAE).

Current Performance: The model predicts fruit age with an MAE of < 2.0 days on the hold-out test set.

Future Roadmap...

Refactor the data ingestion pipeline in prepare_data.py to utilize tf.data.Dataset for lazy loading, allowing the pipeline to scale to >100GB datasets without RAM bottlenecks.

Implement continuous model retraining (CML) triggered automatically when the Wasserstein drift score exceeds the critical threshold.
