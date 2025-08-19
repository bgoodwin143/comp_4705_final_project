# log_data.py
import wandb

# 1. Initialize a new W&B run
run = wandb.init(project="Toxic-Comment-Classification-Final", job_type="upload_data")

# 2. Create a W&B Artifact
# The name is the dataset name, and the type is 'dataset'
raw_data_artifact = wandb.Artifact(
    "raw_toxic_comments_dataset",
    type="dataset",
    description="The raw training data (train.csv) from the Jigsaw Kaggle competition.",
)

# 3. Add the file to the artifact
# This points to the local file
raw_data_artifact.add_file("train.csv")

# 4. Log the artifact to W&B
# This uploads the file and versions it
run.log_artifact(raw_data_artifact)
print("Raw data artifact has been logged to W&B.")

run.finish()
