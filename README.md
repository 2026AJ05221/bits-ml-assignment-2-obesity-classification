# Obesity Level Classification — Machine Learning Assignment 2

## a. Problem statement
Build and compare five supervised classification models to predict the obesity-level category (`NObeyesdad`) from demographic, eating-habit, activity, and transport-related features. The project also provides a Streamlit app that evaluates a saved model against uploaded labelled test data.

> Educational machine-learning project only; outputs are not medical advice or a diagnosis.

## b. Dataset description
- File: `ObesityDataSet_raw_and_data_sinthetic.csv`
- Rows: 2,111; columns: 17 (16 input features + 1 target)
- Target: `NObeyesdad`, a 7-class categorical variable.
- Classes: Insufficient_Weight, Normal_Weight, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III, Overweight_Level_I, Overweight_Level_II
- Input features: Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS
- Dataset includes numeric and categorical fields. Categorical features are one-hot encoded; numeric features are median-imputed and standardized. Categorical values are most-frequent-imputed if needed.
- Split: stratified 80% train / 20% held-out test, `random_state=42`.
- The supplied test file is the held-out 20% split and includes the target label, as required by the assignment for evaluation.

## c. GitHub Repository Link
**Add your GitHub repository URL here after you create and push this project:** "https://github.com/2026AJ05221/bits-ml-assignment-2-obesity-classification"

## d. Models used and evaluation metrics
Metrics are computed on the same held-out test split. AUC is macro-averaged one-vs-rest (OVR) multiclass ROC AUC. Precision, recall, and F1 are weighted averages. Values are specific to this split and random seed; results can differ if the split or hyperparameters change.

| Model | Accuracy | AUC (macro OVR) | Precision (weighted) | Recall (weighted) | F1 (weighted) | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8747 | 0.9834 | 0.8729 | 0.8747 | 0.8731 | 0.8539 |
| Decision Tree | 0.9173 | 0.9559 | 0.9199 | 0.9173 | 0.9178 | 0.9036 |
| K-Nearest Neighbors | 0.8251 | 0.9650 | 0.8248 | 0.8251 | 0.8170 | 0.7981 |
| Gaussian Naive Bayes | 0.5083 | 0.8372 | 0.5275 | 0.5083 | 0.4607 | 0.4471 |
| Support Vector Machine | 0.9243 | 0.9929 | 0.9264 | 0.9243 | 0.9248 | 0.9118 |

### Observations about model performance

| Model | Observation |
|---|---|
| Logistic Regression | Strong baseline with 87.47% accuracy and macro OVR AUC 0.9834. Its linear decision boundary may miss nonlinear class patterns. |
| Decision Tree | High accuracy (91.73%) and MCC (0.9036); easy to interpret, but a single tree can be sensitive to small data changes. |
| KNN | Moderate performance (82.51% accuracy). Scaling is important for distance-based learning; neighbourhood voting can confuse nearby obesity categories. |
| Naive Bayes (Gaussian) | Lowest accuracy (50.83%) and weighted F1 (0.4607) in this experiment. Its conditional-independence and Gaussian assumptions may not fit these features well. |
| Support Vector Machine | Best overall in this fixed split: 92.43% accuracy, 0.9929 macro OVR AUC, 0.9248 weighted F1 and 0.9118 MCC. Scaling and probability estimates are included in the saved pipeline. |
| Overall winner | **Support Vector Machine** has the strongest overall results on this held-out split. Do not interpret this as proof it will be best on every dataset or split. |

## Repository contents
- `app.py` — Streamlit evaluation interface
- `train_models.py` — reproducible training/evaluation script
- `requirements.txt` — Python dependencies
- `README.md` — project and result documentation
- `test_data.csv` — labelled held-out test rows for app demonstration
- `ObesityDataSet_raw_and_data_sinthetic.csv` — provided dataset used to train
- `model/` — saved fitted pipelines for all five classifiers and model metadata
- `model_metadata.json` — input feature names and target classes for app validation

## Run in BITS Virtual Lab
1. Open the BITS Virtual Lab environment and launch its Jupyter Notebook/JupyterLab or terminal interface.
2. Upload/extract this project folder into your workspace. Upload the CSV dataset as well if it is not already in the folder.
3. In a notebook cell, check Python and package availability:
   ```python
   import sys
   print(sys.version)
   %pip --version
   ```
4. Install dependencies in the active notebook kernel:
   ```python
   %pip install -r requirements.txt
   ```
   If the lab does not allow `-r`, run `%pip install pandas scikit-learn streamlit matplotlib seaborn joblib`.
5. Restart the kernel if Jupyter requests it, then run:
   ```python
   %run train_models.py
   ```
   Check that five `.joblib` model files are created and metric rows print in the output.
6. If the pre-trained models supplied in `model/` are present, you can skip retraining for app evaluation. For proof of assignment execution, run the training script in BITS Virtual Lab and take your own screenshot showing the code cell and successful output. **Do not use a fabricated screenshot.**
7. To launch Streamlit, use a terminal in the project folder:
   ```bash
   streamlit run app.py --server.port 8501
   ```
   If the virtual lab does not expose port 8501, use the lab's supported app/port-forwarding instructions. Some virtual lab configurations may allow notebook execution but not public web serving; if so, run training/evaluation in the lab and deploy the app from GitHub/Streamlit Community Cloud separately.

## GitHub setup
1. Create a new, empty GitHub repository under your account.
2. Upload the project files and the entire `model/` directory. Keep `app.py`, `requirements.txt`, and `README.md` at repository root.
3. Commit your own changes with meaningful messages; review and understand the code rather than submitting it blindly.
4. Replace the placeholder GitHub URL above with your actual repository URL.
5. Verify that `test_data.csv` and all five model files are present in the repository.

## Deploy Streamlit Community Cloud
1. Push the complete project to GitHub (default branch is usually `main`).
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app / New app**, select your repository and branch, and set the main file path to `app.py`.
4. Click **Deploy** and wait for build logs to finish. If deployment fails, open the logs and compare missing packages with `requirements.txt`.
5. Open the public app URL. Upload `test_data.csv`, select each model in the dropdown, and confirm metrics, confusion matrix, classification report, and predictions appear.
6. Copy the real GitHub URL and real Streamlit URL into the final submission PDF. These URLs cannot be generated until you deploy under your own account.

## Streamlit app behaviour
- CSV upload and validation for all required feature and target columns.
- Dropdown to select any of the five saved models.
- Accuracy, macro OVR AUC, weighted precision/recall/F1, and MCC.
- Confusion matrix and classification report.
- Downloadable CSV containing predictions and correctness flags.
- The uploaded CSV must be labelled; the app evaluates it and never retrains on it.

## Final submission PDF checklist (follow assignment order)
1. GitHub repository link (replace placeholder with your real URL).
2. Live Streamlit app link (replace placeholder with real deployed URL).
3. Screenshot from BITS Virtual Lab (capture it yourself after running the script).
4. Include this README content in the submitted PDF.
