# Final MLOps Project: Toxic Comment Moderation System

This repository contains a production-grade MLOps system for classifying toxic online comments. The project covers the entire ML lifecycle, from experiment tracking and model versioning to automated deployment and live monitoring on AWS.

## Phase 1: Experimentation and Model Management

This phase covers training a baseline model, tracking the experiments with MLflow, and registering the best-performing model for production use.

### 1.1. Environment Setup

To ensure reproducibility, all dependencies are managed in the `requirements.txt` file.

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/bgoodwin143/comp_4705_final_project
    cd mlops-final-project
    ```

2.  **Create and Activate a Virtual Environment** (Recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### 1.2. Download the Dataset

This project uses the Jigsaw Toxic Comment Classification dataset from Kaggle.

1.  Download the data from: [https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)
2.  Unzip the file and place `train.csv` in the root directory of this project.

### 1.3. Running Experiments with MLflow

The `train.py` script will train a model and log all parameters, metrics, and model artifacts to MLflow.

1.  **Start the MLflow Tracking UI**:
    In a new terminal window, navigate to the project directory and run:
    ```bash
    mlflow ui
    ```
    This will start the tracking server. Keep this terminal running.

2.  **Run the Training Script**:
    In your original terminal, run the training script:
    ```bash
    python train.py
    ```

3.  **View Experiment Results**:
    Open your browser and go to `http://localhost:5000`. You will see the "Toxic Comment Classification" experiment with your latest run.

### 1.4. Registering the Production Model

Follow these steps in the MLflow UI to version the model and promote it to production.

1.  Click into your latest run.
2.  Navigate to the **Artifacts** section. You will see a `model` and a `vectorizer` folder.
3.  Click on the `model` artifact folder, then click the blue **"Register Model"** button.
4.  Select **"Create New Model"** and name it `toxic-comment-classifier`.
5.  Navigate to the **Models** tab from the main menu on the left.
6.  Click on your new `toxic-comment-classifier` model.
7.  Click on **Version 1** and use the **Stage** dropdown to transition it to **Production**.

---