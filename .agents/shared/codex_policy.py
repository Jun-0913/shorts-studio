"""Model configuration for schema-constrained editorial requests."""
import os

def model_arguments():
    arguments = ["--config", 'model_reasoning_effort="high"']
    if os.environ.get("STUDIO_MODEL"):
        arguments += ["--model", os.environ["STUDIO_MODEL"]]
    return arguments
