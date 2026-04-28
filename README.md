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