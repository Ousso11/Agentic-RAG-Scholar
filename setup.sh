#!/bin/bash

# Define the name of the Conda environment
ENV_NAME="agentic-rag-env"

# Create a new Conda environment with Python 3.11
echo "Creating Conda environment '$ENV_NAME' with Python 3.11..."
conda create --name $ENV_NAME python=3.11 -y

# Activate the newly created environment
echo "Activating the environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Check if activation was successful
if [ $? -ne 0 ]; then
  echo "Error: Failed to activate Conda environment. Please try activating it manually: 'conda activate $ENV_NAME'"
  exit 1
fi

# Install the required packages from requirements.txt
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

# Final confirmation
if [ $? -eq 0 ]; then
  echo "✅ Environment setup complete!"
  echo "You can now run your application by simply activating the environment: 'conda activate $ENV_NAME'"
else
  echo "❌ Error: Failed to install packages. Please check your requirements.txt file and try again."
fi