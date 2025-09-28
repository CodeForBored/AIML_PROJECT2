# Model Comparison Report
**Generated:** 2025-09-28T20:32:18.327977 UTC

## Summary Table

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| rf | 0.8028 | 0.1667 | 0.0028 | 0.0056 | 0.5084 |
| svm_linear | 0.4778 | 0.1877 | 0.5043 | 0.2736 | 0.5146 |
| svm_poly | 0.5900 | 0.1991 | 0.3647 | 0.2575 | 0.5021 |
| svm_rbf | 0.5333 | 0.2036 | 0.4786 | 0.2857 | 0.4826 |

## Observations & Insights

- Random Forest: good accuracy but may under-predict rare attrition class when data is imbalanced.
- SVM variants trained with class_weight='balanced' typically improved recall (catching Attrition=Yes).

## Top features from Random Forest

- Monthly_Working_Hours: 0.2273
- Training_Hours_per_Year: 0.1956
- Age: 0.1862
- Years_of_Experience: 0.1770
- Performance_Rating: 0.0735
- Job_Satisfaction_Level: 0.0482
- Promotion_in_Last_2_Years: 0.0170
- Dept_Tech: 0.0158

## Recommendations

- Address class imbalance (SMOTE / resampling).
- Threshold tuning for probability-based models.
- Try ensemble/stacking and add explainability (SHAP).