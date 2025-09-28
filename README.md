# AIML_PROJECT2
Employee Performance &amp; Attrition Prediction using Random Forest and SVM

# Employee Attrition Prediction (AIML Club Project)

## 📌 Overview
This project predicts **employee attrition (Yes/No)** using machine learning models on HR data.  
It demonstrates a complete **AI/ML pipeline**: preprocessing, training, evaluation, and reporting.  
To handle class imbalance, an extra experiment with **SMOTE oversampling** is also included.

---

## 📂 Dataset
File: `Employee_Performance_Retention.csv`  
Columns include:
- **Demographics**: Age, Department, Years of Experience  
- **Work factors**: Monthly Working Hours, Training Hours per Year, Performance Rating  
- **Job context**: Job Satisfaction, Promotion history  
- **Target**: Attrition (Yes / No)  

---

## ⚙️ Setup
Clone this repo, then run:

```bash
# (optional) create venv
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell

# install dependencies
pip install -r requirements.txt

🚀 Usage
1. Baseline pipeline

Run the main script:

python main.py


This will:

Preprocess the dataset

Train models (Random Forest + SVMs)

Evaluate them on a test set

Generate Model_Comparison_Report.md with metrics and insights

2. Extra credit (SMOTE oversampling)

To improve recall for attrition prediction:

python xtra_smote.py


This applies SMOTE to balance the training set and retrains Random Forest + RBF SVM.
Results are saved to smote_summary.csv.

📊 Results (summary)

Random Forest (baseline): High accuracy, very low recall (class imbalance issue)

SVMs (baseline): Lower accuracy, but better recall with class weights

SMOTE (extra): Recall improved significantly → better at catching attrition cases

📁 Repo structure
AIML_Task2/
│── Employee_Performance_Retention.csv   # Dataset
│── main.py                              # Full ML pipeline
│── xtra_smote.py                        # SMOTE oversampling experiment
│── requirements.txt                     # Dependencies
│── README.md                            # Documentation
│── Model_Comparison_Report.md           # Results & insights

