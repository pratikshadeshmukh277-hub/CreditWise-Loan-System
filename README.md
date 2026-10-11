# CreditWise Loan System — Streamlit App

An educational Streamlit dashboard for exploring a loan-approval dataset, comparing Logistic Regression, K-Nearest Neighbors (KNN), and Gaussian Naive Bayes, and generating a model-based estimate for an applicant.

> **Important:** This is a learning/demo project, not a lending decision system. Do not use its predictions as the sole basis for real credit decisions.

## Project files

```text
CreditWise-Loan-System/
├── app.py
├── requirements.txt
├── README.md
├── loan_approval_data.csv   # add your dataset here (recommended)
└── .streamlit/
    └── config.toml
```

The app can also read `loan_prediction.csv`, `train.csv`, or `data/loan_approval_data.csv`. Alternatively, upload a CSV in the sidebar.

## Dataset format

The target column should be named `Loan_Status` and contain at least two classes, for example `Y`/`N`, `Approved`/`Rejected`, or `1`/`0`. The app recognizes common loan dataset columns such as:

- Gender, Married, Dependents, Education, Self_Employed
- ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
- Credit_History, Property_Area, Loan_Status

If your target column has a different name, edit the target mapping section in `app.py`. Avoid committing personal or confidential applicant data to a public repository.

## Run locally on Windows

1. Install Python (use a version supported by Streamlit Community Cloud).
2. Open Command Prompt or PowerShell in this folder.
3. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Put your dataset CSV in the repository root as `loan_approval_data.csv`.
6. Start the dashboard:

   ```powershell
   streamlit run app.py
   ```

7. Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Push the app to GitHub

If you already cloned the repository, copy `app.py`, `requirements.txt`, `README.md`, and `.streamlit/config.toml` into that repository. Add the dataset only if you have permission to share it.

```powershell
git add app.py requirements.txt README.md .streamlit/config.toml
git commit -m "Add Streamlit loan prediction dashboard"
git push
```

If the repository is new/empty and you have not cloned it yet:

```powershell
git clone https://github.com/pratikshadeshmukh277-hub/CreditWise-Loan-System.git
cd CreditWise-Loan-System
```

Then copy the files into this folder and run the commands above. If GitHub shows the repository as empty, make sure you commit and push your notebook, dataset (if shareable), and app files to the correct branch.

## Deploy on Streamlit Community Cloud

1. Push the app files to the GitHub repository's `main` branch.
2. Sign in at https://share.streamlit.io/ using GitHub.
3. Choose **Create app**.
4. Select repository `pratikshadeshmukh277-hub/CreditWise-Loan-System`.
5. Select branch `main`.
6. Set the main file path to `app.py`.
7. Choose an optional app subdomain, then click **Deploy**.
8. If deployment fails, open the app's logs and fix the first Python/dependency error.
9. After it launches, copy the `*.streamlit.app` URL and test it in a private/incognito browser window.

## Notes

- The app trains the models when it starts, so the CSV must contain enough rows and at least two target classes.
- Preprocessing (imputation, one-hot encoding and scaling) is kept inside each model pipeline to avoid fitting transformations on the test set.
- Reported accuracy uses a single 80/20 holdout split and can vary with the dataset. It is not a substitute for full model validation.
- If your notebook uses a different target encoding, feature set, or preprocessing logic, align the app with that original pipeline before presenting results.
