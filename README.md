# Final MLOps Project: Toxic Comment Moderation System

This repository contains a complete, production-grade MLOps system for classifying toxic online comments. The project covers the entire ML lifecycle, from experiment tracking and data versioning with Weights & Biases, to automated CI/CD, to a multi-instance deployment on AWS with live monitoring.

## System Architecture

The final architecture is a multi-component cloud application designed for scalability and maintainability:

*   **Experiment Tracking & Model Registry (Weights & Biases)**: All model training experiments are tracked in W&B. This includes hyperparameters, metrics, code versions, and versioned datasets and model pipeline artifacts. The W&B Model Registry is used to manage model versions and promote them to production.

*   **Backend API (FastAPI on EC2)**: A containerized FastAPI service runs on a dedicated EC2 instance. It automatically downloads and serves the model version currently tagged as "Production" from the W&B registry. It exposes a `/predict` endpoint and logs every transaction to a central DynamoDB table.

*   **Cloud Database (AWS DynamoDB)**: A managed NoSQL database (`prediction_logs` table) acts as the central data store. It logs every prediction from the API, including the input text, the model's classification, a timestamp, and the model version used.

*   **User Frontend (Streamlit on EC2)**: A containerized Streamlit application runs on a separate EC2 instance. It provides a simple UI for users to input text and receive a classification by calling the backend API.

*   **Monitoring Dashboard (Streamlit on EC2)**: A second containerized Streamlit application, also running on the frontend EC2 instance, serves as the monitoring hub. It connects directly to the DynamoDB table to visualize live production data, including the distribution of predictions (target drift) and prediction volume over time.

*   **CI/CD Pipeline (GitHub Actions)**: An automated workflow validates all code changes. On every pull request to the `main` branch, it runs a linter (`ruff`) and the full `pytest` test suite, ensuring code quality and preventing regressions.

## Phase 1: Experimentation with Weights & Biases

This phase covers setting up the environment, versioning the dataset, and running training experiments.

### 1.1 Environment Setup
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/[your-github-username]/comp_4705_final_project.git
    cd comp_4705_final_project
    git checkout dev
    ```
2.  **Create and Activate a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Log in to W&B**:
    ```bash
    wandb login
    ```
    (You will need a free account and your API key from wandb.ai/settings)

### 1.2 Data and Model Versioning
The project uses W&B Artifacts to version both the dataset and the trained model pipeline.

1.  **Download the Dataset**: Download `train.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data) and place it in the project's root directory.

2.  **Log the Dataset Artifact**: Run the `log_data.py` script once to upload and version the dataset in W&B.
    ```bash
    python log_data.py
    ```
3.  **Run Training and Log the Model**: Execute the `train.py` script. It will download the versioned dataset, train the model, and upload the final `toxic-comment-pipeline` artifact to W&B.
    ```bash
    python train.py
    ```
4.  **Promote the Model in the W&B Registry**:
    *   Navigate to your project's "Registry" tab on the W&B website.
    *   Create a new model registry named `toxic-comment-classifier`.
    *   Find your `toxic-comment-pipeline` artifact and link it to the registry.
    *   Add the **`production`** alias to the version you want to deploy.

## Phase 2-5: Deployment to AWS

This section provides a complete guide for deploying the entire application stack to AWS.

### Prerequisites
*   An AWS account with permissions to create EC2 instances, IAM roles, and DynamoDB tables.
*   The AWS CLI configured locally (optional but helpful).

### Step 1: Create Cloud Resources (IAM & DynamoDB)
1.  **Create DynamoDB Table**:
    *   In the AWS Console, navigate to DynamoDB.
    *   Create a new table named **`prediction_logs`**.
    *   Set the **Partition key** to `prediction_id` (Type: String).
    *   Ensure the table is created in the region you intend to deploy to (e.g., `us-east-2`).

2.  **Create IAM Role**:
    *   In the IAM service, create a new role for an **AWS service (EC2)**.
    *   Select EC2!! Not dynamoDB
    *   Attach the `AmazonDynamoDBFullAccess` permission policy.
    *   Name the role **`EC2-DynamoDB-Access-Role`**.

### Step 2: Create and Configure EC2 Security Groups
1.  **Create `backend-sg`**:
    *   Create a security group for the backend.
    *   Add an **inbound rule** for **SSH (Port 22)** from `My IP`.
2.  **Create `frontend-sg`**:
    *   Create a security group for the frontend.
    *   Add an **inbound rule** for **SSH (Port 22)** from `My IP`.
    *   Add an **inbound rule** for **Custom TCP (Port 8501)** from `Anywhere-IPv4`.
    *   Add an **inbound rule** for **Custom TCP (Port 8502)** from `Anywhere-IPv4`.
3.  **Link the Security Groups**:
    *   Edit the inbound rules for **`backend-sg`**.
    *   Add a rule for **Custom TCP (Port 8000)** and set the **Source** to the ID of your `frontend-sg`.

### Step 3: Launch and Deploy EC2 Instances

1.  **Launch the Backend Server**:
    *   Launch a `t2.micro` EC2 instance with Ubuntu.
    *   Call it backend
    *   create or download an exisitng key pair 
    *   During launch, assign the **`backend-sg`** security group.
    *   In "Advanced details," attach the **`EC2-DynamoDB-Access-Role`** IAM instance profile.
    *   SSH into the instance, install Git and Docker, and configure the Docker group.
    *   Somewhat optional: change the permissions on .pem i.e., chmod   400 on the filename 
    *   SSH into the instance: i.e., ssh -i "key.pem" ubuntu@ec2ip
    *   Run: sudo apt-get update -y
    *   Run: sudo apt-get install git -y
    *   Run: sudo apt-get install docker.io -y
    *   Run: sudo usermod -aG docker ${USER}
    *   Then log out of the machine and log back in
    *   Clone the repository: `https://github.com/bgoodwin143/comp_4705_final_project.git`
    *   cd comp_4705_final_project
    *   Checkout dev branch
    *   Build the backend image: `docker build -t fastapi-backend -f api/Dockerfile .`
    *   Run the backend container, injecting your W&B API Key as a secret:
        ```bash
        docker run -d --name api_container \
          -p 8000:8000 \
          -e WANDB_API_KEY="<YOUR_WANDB_API_KEY>" \
          fastapi-backend
        ```

2.  **Launch the Frontend Server**:
    *   Launch a second `t2.micro` EC2 instance with Ubuntu.
    *   Call it frontend
    *   Assign the **`frontend-sg`** security group.
    *   Attach the **`EC2-DynamoDB-Access-Role`** IAM instance profile.
    *   SSH in, install Git and Docker, and configure the Docker group.
    *   Get the **Private IP Address** of your backend server from the EC2 dashboard.
        *   Somewhat optional: change the permissions on .pem i.e., chmod   400 on the filename 
    *   SSH into the instance: i.e., ssh -i "key.pem" ubuntu@ec2ip
    *   Run: sudo apt-get update -y
    *   Run: sudo apt-get install git -y
    *   Run: sudo apt-get install docker.io -y
    *   Run: sudo usermod -aG docker ${USER}
    *   Then log out of the machine and log back in
    *   Clone the repository: `https://github.com/bgoodwin143/comp_4705_final_project.git`
    *   cd comp_4705_final_project
    *   Checkout dev branch
    *   Build the frontend images:
        ```bash
        docker build -t streamlit-frontend -f frontend/Dockerfile .
        docker build -t streamlit-monitoring -f monitoring/Dockerfile .
        ```
    *   Run the frontend containers, injecting the backend's private IP:
        ```bash
        # User-facing app
        docker run -d --name frontend_container -p 8501:8501 \
          -e API_URL="http://<BACKEND_PRIVATE_IP>:8000" \
          streamlit-frontend

        # Monitoring dashboard
        docker run -d --name monitoring_container -p 8502:8502 streamlit-monitoring
        ```

### Step 4: Access the Live Application (I used a pesonal EC2 account and left my tests live)
*   **User Interface**: `http://3.18.221.12:8501//:8501/`
*   **Monitoring Dashboard**: `http://3.18.221.12:8502/`

## Project Links
*   **W&B Project Dashboard**: (https://wandb.ai/bensharn-university-of-denver/Toxic-Comment-Classification-Final?nw=nwuserbensharn)