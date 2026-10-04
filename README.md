# Credit Card Fraud Detection
## Project Overview
This project uses machine learning to identify fraudulent credit card transactions.
The aim was to build and compare diffrent classification models and evaluate how well the detect fraud.
Because fraudulent transactions are much rarer than normal transactions, I focused on metrics such as precision, recall and F1-score rather than accuracy alone.

## Dataset
Source: Kaggle Credit Card Fraud Detection dataset.
The dataset contains 284, 807 credit card transactions.
Each transaction contains:
- 'Time'
- 'V1' to 'V28'
- 'Amount'
- 'Class'

The 'Class' column is the target:
- '0' = normal transaction
- '1' = fraudulent transaction

The dataset is highly imbalanced, with only 492 fraudulent transactions.

## Tools Used
- Python
- pandas
- scikit-learn
- Logistic Regression
- Random Forest
- StandardScaler

## Project Steps
1. Loaded the credit card transaction dataset using pandas.
2. Explored the dataset and checked for missing values.
3. Checked the number of normal and fraudulent transactions.
4. Separated the input features from the target variable.
5. Split the data into 80% training and 20% test data.
6. Used stratification to keep a similar fraud ratio in both sets.
7. Scaled the 'Time' and 'Amount' features using StandardScaler
8. Trained a Logistic Regression model.
9. Evaluated the model using precision, recall, F1-score and a confusion matrix.
10. Trained a Random Forest model and compared its performance with Logistic Regression.

## Model Results

### Logistic Regression
Fraud class results:

- Precision: 0.83
- Recall: 0.64
- F1-score: 0.72

Confusion matrix:

- True Negatives: 56, 851
- False Positives: 13
- False Negatives: 35
- True Positives: 63

The Logistic Regression model correctly detected 63 of the 98 fraudulent transactions in the test set.

### Random Forest
Fraud class results:

- Precision: 0.94
- Recall: 0.82
- F1-score: 0.87

Confusion matrix:
- True Negatives: 56, 859
- False Positives: 5
- False Negatives: 18
- True Positives: 80

The Random Forest model correctly detected 80 of the 98 fraudulent transactions.

## Conclusion
Random Forest performed better than Logistic Regression.
It achieved higher precision, recall and F1-score and detected more fraudulent transactions while producing fewer false positives.
For this dataset, Random Forest was therefore the better-performing model.

## Future Improvements
Future versions of the project could include:
- testing additional machine learning models
- tuning model parameters
- dealing with class imbalance using additional techniques
- creating an API for fraud predictions
- deploying the model
- adding model monitoring
