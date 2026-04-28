# Nepal Irrigation Need Prediction

This project applies machine learning to predict irrigation need levels in Nepal as a 3-class classification problem:
- Low
- Medium
- High

## Models Used
- Logistic Regression
- Decision Tree
- Random Forest

## Experiments
1. Baseline models (without SMOTE)
2. SMOTE-based models
3. Final model selection based on macro F1-score

## Project Structure
- `data/raw/` : training and test datasets
- `src/` : source code
- `outputs/figures/` : confusion matrices
- `outputs/metrics/` : model summaries and reports
- `outputs/models/` : saved final model
- `outputs/predictions/` : final test predictions
- `notebooks/` : notebook workflow for demonstration/submission

## Setup
Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt

## Dataset
The raw dataset files are not included in this repository due to GitHub file size limits.

Please place the following files inside `data/raw/` before running the project:

- `nepal_dataset_v2_train.csv`
- `nepal_dataset_v2_test.csv`

## Trained Model
The saved trained model file is not included in this repository due to GitHub file size limits.  
You can regenerate it by running the training scripts.