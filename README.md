# Telco Customer Churn Classification

Predicting which telecom customers will churn, comparing five scikit-learn classifiers, with a Streamlit app that loads the saved models and evaluates them on an uploaded CSV without retraining.

**GitHub repository:** https://github.com/abhijaain-cell/telco-customer-churn-classifier

**Live app:** https://telco-customer-churn-classifier-abhi.streamlit.app/

## 1. Problem statement

A telecom operator loses revenue every time a subscriber leaves (churns). Retention offers work best when they reach the right customers before they leave, so the goal is to predict, from a customer's contract, services and billing details, whether that customer will churn.

This is a **binary classification** problem with **Churn = Yes (1)** as the positive class. Five classifiers were trained on the same stratified train/test split and compared on six metrics: Accuracy, AUC, Precision, Recall, F1 and the Matthews Correlation Coefficient (MCC).

## 2. Dataset description

| Item | Value |
|---|---|
| Source | [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), IBM sample data. The raw dataset is not included in this repository. |
| File | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Instances | 7,043 customers, one row each, 21 columns; no duplicate customer IDs and no duplicate rows |
| Identifier | `customerID` (not used for prediction) |
| Features | 19: 16 categorical and 3 numeric |
| Categorical features | `gender`, `SeniorCitizen` (stored as 0/1), `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Numeric features | `tenure` (months), `MonthlyCharges`, `TotalCharges` |
| Target | `Churn`: Yes = 1 (churned), No = 0 |
| Class counts | No 5,174 (73.46%), Yes 1,869 (26.54%); imbalance ratio 2.77 |
| Missing values | No NaN values, but `TotalCharges` is stored as text and contains 11 blank cells (a single space) |

Findings from the exploratory analysis:

- All 11 blank `TotalCharges` values belong to customers with `tenure` = 0 and Churn = No: new customers who had not yet been billed.
- `TotalCharges` is almost exactly `tenure × MonthlyCharges` (ratio median 1.000, range 0.689 to 1.573), so it is largely redundant but not an exact copy.
- 40 rows are identical to another row on all 19 features but belong to different customers (all with `tenure` = 1). In 18 of these groups the customers have different churn outcomes, which shows the features cannot fully explain churn.
- No column copies the label; the strongest numeric association with churn is `tenure` (Spearman −0.367).
- Churn rate varies strongly by category: month-to-month contracts 42.7% against 2.8% for two-year contracts; fibre-optic internet 41.9% against 19.0% for DSL; electronic-check payment 45.3% against 15.2% to 19.1% for the other methods.

## 3. Preprocessing methodology

1. **Identifier removed:** `customerID` was dropped, because it identifies a person and carries no predictive meaning.
2. **`TotalCharges` cleaned:** converted from text to numbers; the 11 blanks were set to 0, because these customers had never been billed.
3. **Target encoded:** Churn Yes = 1, No = 0.
4. **Rows kept:** the 40 feature-identical rows are different customers, not duplicates, so all 7,043 rows were kept.
5. **Stratified split:** 80/20 with `random_state = 42`, giving 5,634 training rows and 1,409 test rows, both with 26.54% churn. The test rows are saved as `models/test_set.csv`.
6. **Encoding and scaling inside each pipeline:** a `ColumnTransformer` one-hot encodes the 16 categorical columns (`OneHotEncoder` with `drop="if_binary"` and `handle_unknown="ignore"`), giving 40 model inputs, and transforms the 3 numeric columns per model:
   - `StandardScaler` for Logistic Regression, KNN and SVM
   - `PowerTransformer` (Yeo–Johnson) for Gaussian Naive Bayes
   - no transformation for the Decision Tree
7. **No leakage:** the encoder and scalers are part of each saved pipeline and were fitted on the training rows only. A validation check confirmed that the saved scaler holds the training-set means, not the full-data means, and that no customer appears in both the training and the test set.
8. **Class imbalance:** `class_weight="balanced"` is used for Logistic Regression, Decision Tree and SVM.

**Metrics:** Precision, Recall and F1 are calculated for the **Churn** class, the customers the business wants to find. AUC is computed from the predicted churn probability, and MCC summarises the whole 2 × 2 confusion matrix. Accuracy alone is misleading here: predicting "No churn" for every customer already scores 73.46%.

## 4. Classifiers

All five models are full scikit-learn pipelines (preprocessing + classifier), trained with scikit-learn 1.8.0.

| Classifier | How it works | Configuration |
|---|---|---|
| Logistic Regression | A linear model that converts a weighted sum of the inputs into a churn probability with the logistic (sigmoid) function. | `C=1.0`, `max_iter=5000`, `class_weight="balanced"`, lbfgs solver; numeric columns standardised |
| Decision Tree | Splits the customers with a sequence of if/else questions on the features, choosing each split to make the groups as pure as possible. | Gini criterion, no depth limit (unpruned), `class_weight="balanced"`; raw numeric values |
| K-Nearest Neighbors | Labels a customer by majority vote of the 5 most similar training customers, using Euclidean distance. | `n_neighbors=5`, uniform weights; numeric columns standardised |
| Gaussian Naive Bayes | Applies Bayes' theorem, assuming every input follows a normal distribution within each class and that inputs are independent given the class. | `var_smoothing=1e-9`; numeric columns power-transformed (Yeo–Johnson) |
| Support Vector Machine | Finds a maximum-margin boundary between the classes; the RBF kernel allows a curved boundary. Probabilities are added by calibration. | RBF kernel, `C=1.0`, `gamma="scale"`, `class_weight="balanced"`; wrapped in `CalibratedClassifierCV` (sigmoid, 5-fold); numeric columns standardised |

## 5. Results: comparison of all five models

Results on the 1,409 held-out test customers (374 churners, 1,035 non-churners):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:--|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 0.7381 | 0.8415 | 0.5043 | 0.7834 | 0.6136 | 0.4531 |
| Decision Tree | 0.7360 | 0.6566 | 0.5028 | 0.4866 | 0.4946 | 0.3160 |
| KNN | 0.7622 | 0.7766 | 0.5543 | 0.5321 | 0.5430 | 0.3825 |
| Gaussian Naive Bayes | 0.6849 | 0.8128 | 0.4499 | 0.8396 | 0.5858 | 0.4138 |
| Support Vector Machine | 0.7821 | 0.8219 | 0.5827 | 0.6310 | 0.6059 | 0.4564 |

Confusion-matrix counts behind these metrics (TP = churners correctly flagged, FN = churners missed, FP = non-churners flagged):

| ML Model Name | Correct (of 1,409) | TP | FN | FP | TN | Customers flagged |
|:--|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 1,040 | 293 | 81 | 288 | 747 | 581 |
| Decision Tree | 1,037 | 182 | 192 | 180 | 855 | 362 |
| KNN | 1,074 | 199 | 175 | 160 | 875 | 359 |
| Gaussian Naive Bayes | 965 | 314 | 60 | 384 | 651 | 698 |
| Support Vector Machine | 1,102 | 236 | 138 | 169 | 866 | 405 |

## 6. Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | The best AUC (0.8415) and F1 (0.6136), and an MCC (0.4531) almost equal to the SVM. Class weighting makes it catch 293 of the 374 churners (recall 0.7834), but it flags 581 customers, 288 of whom do not churn (precision 0.5043). Its coefficients are easy to explain: longer `tenure` (−1.14) and a two-year contract (−0.81) are the strongest signs a customer will stay, while fibre-optic internet (+0.67) and a month-to-month contract (+0.62) push towards churn. |
| Decision Tree | The weakest model: AUC 0.6566, F1 0.4946 and MCC 0.3160. It finds only 182 of the 374 churners (recall 0.4866), and its accuracy (0.7360) is barely above the 73.46% of always predicting "No churn". The unpruned tree grew to depth 26 with 1,250 leaves for 5,634 training rows, so it memorises individual customers, and its leaves give almost only 0/1 probabilities, which hurts AUC. |
| KNN | Accuracy 0.7622 but recall only 0.5321 (199 of 374 churners) and MCC 0.3825. KNN has no class weighting, so the majority "No churn" class dominates the neighbourhood votes. Distances over 40 mostly one-hot inputs are also a weak notion of similarity. The saved model stores all 5,634 training rows, which makes it the largest model file. |
| Gaussian Naive Bayes | The highest recall (0.8396, 314 of 374 churners), but it flags 698 customers, 384 of whom do not churn, giving the lowest precision (0.4499) and accuracy (0.6849). The one-hot 0/1 inputs do not follow a bell curve, and the "No internet service" value repeated across six columns breaks the independence assumption. Its AUC (0.8128) is still reasonable, so its ranking of customers is better than its yes/no decisions. |
| Support Vector Machine | The best Accuracy (0.7821), Precision (0.5827) and MCC (0.4564): 1,102 of 1,409 customers correct. It finds 236 of the 374 churners (recall 0.6310) with 169 false alarms, the most balanced trade-off of the five models. Its AUC (0.8219) and F1 (0.6059) are second only to Logistic Regression. |

## 7. Overall best-performing model

**Support Vector Machine.**

- It has the best **Accuracy (0.7821)**, **Precision (0.5827)** and **MCC (0.4564)**, and is second on **AUC (0.8219)** and **F1 (0.6059)**.
- MCC is the most reliable single metric for an imbalanced binary problem, because it uses all four cells of the confusion matrix; the SVM leads on it.
- It gives the most balanced trade-off: 236 churners found with 169 false alarms, compared with 288 false alarms for Logistic Regression and 384 for Gaussian Naive Bayes.

**Logistic Regression is a very close second** (MCC 0.4531), with the best AUC and F1 and a much higher recall (0.7834). The better choice depends on the cost of a retention action: for cheap offers, such as a discount SMS, Logistic Regression catches more churners; for expensive ones, such as a call from a retention agent, the SVM wastes fewer contacts.

All five models score noticeably lower than on cleaner datasets, because churn depends on factors the data does not record, as the 18 groups of identical customers with different outcomes show.

**Limitation:** for churned customers, `tenure` is their full lifetime with the operator, while for current customers it is their tenure so far, so part of the strong tenure effect comes from how this snapshot was taken.

## 8. Running the Streamlit application

**Online:** open the live app link at the top of this page.

**Locally:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

The saved models require **scikit-learn 1.8.0**, the version used for training; `requirements.txt` pins it. When deploying on Streamlit Community Cloud, choose Python 3.12 under Advanced settings.

**Using the app:**

1. **Sidebar:** choose one of the five models, then upload a CSV or tick **Use the bundled test set**. The bundled `test_set.csv` can be downloaded from the sidebar.
2. **Predict & evaluate tab:** the six metrics for the chosen model, a 2 × 2 confusion matrix, a classification report, and a predictions table with the churn probability, downloadable as CSV.
3. **Model comparison tab:** the metrics table for all five models, with the best value in each column highlighted.
4. **About the data tab:** dataset, target, preprocessing and the list of required input columns.

**Input CSV format:** the 19 customer columns listed in section 2. Extra columns such as `customerID`, and a different column order, are fine, so the original Kaggle CSV can be uploaded directly. The app applies the same cleaning as training: blank `TotalCharges` values become 0 (with a message), `SeniorCitizen` may be 0/1 or Yes/No, and `Churn` labels may be Yes/No or 1/0. Without a `Churn` column the app shows predictions only. Category values never seen in training are reported, and a missing column, an empty file or a file that is not a CSV shows an error message. If the labels contain only one class, AUC shows n/a. The app only loads the saved models; it never trains.

## 9. Repository structure

```
telco-customer-churn-classifier/
├── app.py                          Streamlit app (loads saved models, never retrains)
├── requirements.txt                Dependencies, pinned to scikit-learn 1.8.0
├── README.md                       This file
├── .gitignore                      Excludes caches, credentials and the raw data folder
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── gaussian_nb.joblib
│   ├── svm.joblib                  Each file = full pipeline (encoding + scaling + model)
│   ├── metadata.json               Feature order, column types, class names, training settings
│   ├── test_set.csv                1,409 held-out customers (19 features + Churn)
│   └── metrics_project_b.csv       The 30-value metrics table in section 5
└── notebooks/
    └── project_b_notebook.ipynb    EDA, preprocessing, training, evaluation, saving and validation
```
