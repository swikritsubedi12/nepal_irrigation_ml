# A Machine Learning Approach to Crop-Specific Irrigation Need Prediction in Nepal Using a Random Forest Classifier

This project predicts **crop-specific irrigation need levels in Nepal** as **Low, Medium, or High** using machine learning classification models. It compares **Logistic Regression**, **Decision Tree**, and **Random Forest**, and also evaluates model performance **before and after SMOTE-based class imbalance handling**.

The final selected model is the **baseline Random Forest classifier**.

---

## Project Summary

Agriculture in Nepal is strongly influenced by soil condition, weather, crop type, and irrigation history. This project applies machine learning to predict irrigation need levels using agricultural data related to soil properties, climate conditions, crop characteristics, irrigation history, and regional context.

The target variable is:
- **Low**
- **Medium**
- **High**

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- scikit-learn
- imbalanced-learn
- Matplotlib
- Seaborn

---

## Project Structure

```text
nepal-irrigation-ml/
│
├── data/
│   └── raw/
├── notebooks/
│   └── 01_main_workflow.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── predictions/
│   └── models/
├── src/
│   ├── compare_results.py
│   ├── config.py
│   ├── data_utils.py
│   ├── evaluate.py
│   ├── feature_importance.py
│   ├── models.py
│   ├── predict_test.py
│   ├── preprocessing.py
│   ├── run_baseline.py
│   └── run_smote.py
├── README.md
├── requirements.txt
└── .gitignore
Dataset

The raw dataset files are not included in this repository due to GitHub file size limits.

Please place the following files inside:

data/raw/

Required files:

nepal_dataset_v2_train.csv
nepal_dataset_v2_test.csv
Trained Model

The saved trained model file is not included in this repository due to GitHub file size limits.

You can regenerate it by running the training scripts.

Installation

Install the required packages using:

pip install -r requirements.txt
Comprehensive Instructions for Successful Execution

This project was developed in Python using Jupyter Notebook and modular Python scripts.

Option 1: Run Using Jupyter Notebook

Start Jupyter Notebook:

jupyter notebook

Then open:

notebooks/01_main_workflow.ipynb

Run all cells in order from top to bottom.

The notebook demonstrates:

data loading
data inspection
preprocessing
feature engineering
model training
baseline evaluation
SMOTE evaluation
model comparison
output visualization
Option 2: Run Using Python Scripts

Run the project step by step in this order:

python -m src.run_baseline
python -m src.run_smote
python -m src.compare_results
python -m src.feature_importance
python -m src.predict_test
Step 1: Run baseline models
python -m src.run_baseline

This script:

loads the training data
applies preprocessing and feature engineering
trains Logistic Regression, Decision Tree, and Random Forest
evaluates them on the validation set
saves reports and confusion matrices in outputs/
Step 2: Run SMOTE-based models
python -m src.run_smote

This script:

applies SMOTE-based class imbalance handling
retrains the same three models
evaluates them again
saves reports and results in outputs/
Step 3: Compare baseline and SMOTE results
python -m src.compare_results

This script:

combines baseline and SMOTE results
creates a comparison summary
saves the final comparison CSV in:
outputs/metrics/all_model_comparison.csv
Step 4: Extract feature importance from the final model
python -m src.feature_importance

This script:

loads the final Random Forest model
extracts feature importances
saves:
outputs/metrics/top5_feature_importance.csv
outputs/figures/top5_feature_importance.png
Step 5: Generate final test predictions
python -m src.predict_test

This script:

uses the selected final model
predicts irrigation need for the unseen test dataset
saves predictions in:
outputs/predictions/final_test_predictions.csv
Output Files

After successful execution, outputs will be generated in:

outputs/figures/
confusion matrices
class distribution figure
feature importance figure
outputs/metrics/
classification reports
baseline summary
SMOTE summary
model comparison
feature importance CSV
outputs/predictions/
final test predictions
Final Selected Model

The final selected model is:

Baseline Random Forest

It achieved the best overall performance in terms of macro F1-score and class-wise balance.

Evaluation Metrics

The models were evaluated using:

Accuracy
Macro Precision
Macro Recall
Macro F1-score
Confusion Matrix

Macro F1-score was treated as the main comparison metric because it gives equal importance to all three classes, including the minority class.

Engineered Features

Additional engineered features used in the project include:

Moisture Deficit
Soil Stress Score
Temperature–Rainfall Ratio
Irrigation per Hectare
Models Used

The project compares the following classification models:

Logistic Regression
Decision Tree
Random Forest

These models were selected to compare:

a linear baseline model
a single tree-based model
an ensemble tree-based model
Notes for Successful Execution
Make sure the dataset files are placed correctly before running the code.
Do not rename folders or files unless you also update the file paths inside the code.
Run the scripts in the correct order if you want all outputs to be generated properly.
If the trained model file is missing, run the training scripts again to regenerate it.
If using Jupyter Notebook, run all cells from top to bottom in sequence.