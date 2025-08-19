import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import wandb
import joblib
import os
import ast  # Used to safely evaluate string representations of Python literals


def train():
    """
    This function trains a model pipeline, logs metrics and parameters to W&B,
    and saves the final pipeline as a W&B artifact.
    It's designed to be called directly or by a W&B sweep agent.
    """
    # --- 1. Initialize W&B Run ---
    # If called by a sweep agent, the config will be pre-filled.
    # If run directly, it will use the defaults in the 'config' dictionary.
    print("Initializing W&B run...")
    wandb.init(
        project="Toxic-Comment-Classification-Final",
        config={
            # Default hyperparameters for a single run
            "vectorizer": "TfidfVectorizer",
            "model": "LogisticRegression",
            "pipeline_version": 1.0,
            "ngram_range": "(1, 2)",  # Pass tuple as string for YAML compatibility
            "max_features": 15000,
            "C": 1.0,
            "solver": "liblinear",
            "dataset_sample_size": 15000,
            "dataset_artifact": "raw_toxic_comments_dataset:latest",
        },
    )

    # W&B's config object is a dictionary-like object that holds our hyperparameters
    config = wandb.config
    print("W&B run initialized with config:")
    print(dict(config))

    # --- 2. Load Data from W&B Artifact ---
    # This ensures that the training run is tied to a specific version of the dataset.
    print(f"Loading data from W&B Artifact: {config.dataset_artifact}...")
    try:
        raw_data_artifact = wandb.use_artifact(config.dataset_artifact)
        data_dir = raw_data_artifact.download()
        data_path = os.path.join(data_dir, "train.csv")
        df = pd.read_csv(data_path).sample(config.dataset_sample_size, random_state=42)
        print("Data loaded and sampled successfully.")
    except Exception as e:
        wandb.alert(
            title="Failed to load data artifact",
            text=f"Could not load artifact {config.dataset_artifact}. Error: {e}",
        )
        print(f"Error loading data artifact: {e}")
        return  # Exit if data can't be loaded

    # Preprocessing
    df["toxic"] = (
        df[
            ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        ].sum(axis=1)
        > 0
    ).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        df["comment_text"], df["toxic"], test_size=0.2, random_state=42
    )
    print("Data preprocessed and split.")

    # --- 3. Create and Train the Scikit-learn Pipeline ---
    # The pipeline encapsulates the vectorizer and the model.
    print("Building and training pipeline...")
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    # We parse the string from config back into a real tuple
                    ngram_range=ast.literal_eval(config.ngram_range),
                    max_features=config.max_features,
                ),
            ),
            ("logreg", LogisticRegression(C=config.C, solver=config.solver)),
        ]
    )
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # --- 4. Evaluate and Log Metrics ---
    print("Evaluating model and logging metrics to W&B...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    wandb.log({"accuracy": accuracy, "f1_score": f1})
    print(f"Metrics logged: Accuracy={accuracy:.4f}, F1-Score={f1:.4f}")

    # --- 5. Save and Log the Model Pipeline as a W&B Artifact ---
    # This versions our final model object for use in production.
    print("Saving model artifact to W&B...")
    model_artifact_name = "toxic-comment-pipeline"
    model_filename = f"{model_artifact_name}.pkl"
    joblib.dump(pipeline, model_filename)

    artifact = wandb.Artifact(
        name=model_artifact_name,
        type="model",
        description="A TF-IDF and Logistic Regression pipeline for toxic comment classification.",
        metadata=dict(config),  # Associate hyperparameters with the model artifact
    )
    artifact.add_file(model_filename)
    wandb.log_artifact(artifact)
    print("Model artifact logged successfully.")

    # --- 6. Finish the Run ---
    wandb.finish()
    print("\nRun finished! All data has been saved to your W&B project.")


if __name__ == "__main__":
    train()
