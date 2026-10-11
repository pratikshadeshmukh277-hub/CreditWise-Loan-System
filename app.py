import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone

st.set_page_config(
    page_title="CreditWise | Loan Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main {background-color: #f7f9fc;}
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.35rem 1.5rem; border-radius: 18px;
        background: linear-gradient(120deg, #102a43, #176b87);
        color: white; margin-bottom: 1.2rem;
    }
    .hero h1 {color: white; margin: 0; font-size: 2.1rem;}
    .hero p {color: #e7f4fa; margin: .45rem 0 0 0;}
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e5eaf1;
        padding: 14px 16px; border-radius: 14px;
    }
    .small-note {color: #607080; font-size: .88rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>💳 CreditWise Loan Intelligence</h1>
  <p>Explore applicant data, compare classification models, and estimate a loan outcome.</p>
</div>
""", unsafe_allow_html=True)

# ---- Dataset handling ----
DATA_CANDIDATES = [
    Path("loan_approval_data.csv"),
    Path("loan_prediction.csv"),
    Path("train.csv"),
    Path("data/loan_approval_data.csv"),
]

def standardize_columns(df):
    """Map common Kaggle loan-prediction column spellings to consistent names."""
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    aliases = {
        "Loan_ID": "Loan_ID",
        "loan_id": "Loan_ID",
        "Gender": "Gender", "gender": "Gender",
        "Married": "Married", "married": "Married",
        "Dependents": "Dependents", "dependents": "Dependents",
        "Education": "Education", "education": "Education",
        "Self_Employed": "Self_Employed", "self_employed": "Self_Employed",
        "ApplicantIncome": "ApplicantIncome", "Applicant_Income": "ApplicantIncome",
        "CoapplicantIncome": "CoapplicantIncome", "Coapplicant_Income": "CoapplicantIncome",
        "LoanAmount": "LoanAmount", "Loan_Amount": "LoanAmount",
        "Loan_Amount_Term": "Loan_Amount_Term", "LoanAmountTerm": "Loan_Amount_Term",
        "Credit_History": "Credit_History", "CreditHistory": "Credit_History",
        "Property_Area": "Property_Area", "PropertyArea": "Property_Area",
        "Loan_Status": "Loan_Status", "Loan_Status_": "Loan_Status",
        "Loan_Status_(Target)": "Loan_Status", "loan_status": "Loan_Status",
        "LoanApprovalStatus": "Loan_Status", "Loan_Approval_Status": "Loan_Status",
        # Column names used in the provided 1,000-row dataset
        "Loan_Approved": "Loan_Status",
        "Applicant_Income": "ApplicantIncome",
        "Coapplicant_Income": "CoapplicantIncome",
        "Loan_Amount": "LoanAmount",
        "Loan_Term": "Loan_Amount_Term",
    }
    rename = {c: aliases[c] for c in df.columns if c in aliases and c != aliases[c]}
    df = df.rename(columns=rename)
    # Common alternative target names, including case-insensitive variants.
    if "Loan_Status" not in df.columns:
        target_aliases = {
            "approved", "approval_status", "loanapproved",
            "loan_approval", "loan_approved", "loanapproval",
        }
        for col in df.columns:
            if str(col).strip().lower() in target_aliases:
                df = df.rename(columns={col: "Loan_Status"})
                break
    return df

def find_dataset():
    for path in DATA_CANDIDATES:
        if path.exists():
            try:
                return pd.read_csv(path), str(path)
            except Exception:
                pass
    return None, None

@st.cache_data(show_spinner=False)
def load_csv_bytes(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))

with st.sidebar:
    st.header("⚙️ Project controls")
    st.caption("Dataset")
    uploaded = st.file_uploader("Upload your loan CSV", type=["csv"])
    default_df, default_path = find_dataset()
    if uploaded is not None:
        try:
            raw_df = load_csv_bytes(uploaded.getvalue())
            source_name = uploaded.name
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()
    elif default_df is not None:
        raw_df, source_name = default_df, default_path
    else:
        raw_df, source_name = None, None
        st.warning("Add loan_approval_data.csv to the repository root, or upload a CSV here.")
    st.divider()
    st.caption("Target column: Loan_Status (also accepts Loan_Approved)")
    st.caption("Common target values: Yes/No, Y/N, Approved/Rejected, 1/0")

if raw_df is None:
    st.info("**Getting started:** Upload your loan dataset in the sidebar to preview the dashboard. For permanent deployment, commit your CSV to the GitHub repository (only if you are permitted to share it).")
    st.markdown("""
    ### Required project files
    - `app.py` — Streamlit dashboard
    - `requirements.txt` — deployment dependencies
    - `loan_approval_data.csv` — training dataset (or upload one in the sidebar)
    - `README.md` — setup and deployment instructions
    """)
    st.stop()

df = standardize_columns(raw_df)
st.sidebar.success(f"Loaded: {source_name}")
st.sidebar.write(f"{len(df):,} rows · {len(df.columns)} columns")

TARGET = "Loan_Status"
if TARGET not in df.columns:
    st.error("Target column not found. Rename your dataset's approval/rejection column to `Loan_Status`, or update the target mapping in app.py.")
    st.write("Columns found:", list(df.columns))
    st.stop()

df = df.dropna(subset=[TARGET]).copy()
if df.empty:
    st.error("The dataset has no non-empty values in the Loan_Status target column.")
    st.stop()

# Normalize target labels. This remains a classification demo, not a credit decision.
target_values = df[TARGET].astype(str).str.strip()
target_map = {
    "Y": "Approved", "Yes": "Approved", "yes": "Approved",
    "1": "Approved", "1.0": "Approved", "Approved": "Approved",
    "approved": "Approved", "True": "Approved", "true": "Approved",
    "N": "Rejected", "No": "Rejected", "no": "Rejected",
    "0": "Rejected", "0.0": "Rejected", "Rejected": "Rejected",
    "rejected": "Rejected", "False": "Rejected", "false": "Rejected",
}
normalized = target_values.map(target_map).fillna(target_values)
if normalized.nunique() < 2:
    st.error("The target must contain at least two classes (for example, Approved and Rejected).")
    st.stop()
df[TARGET] = normalized

# Exclude identifiers and target from predictors
drop_cols = [TARGET]
for col in ["Loan_ID", "LoanId", "ID", "id"]:
    if col in df.columns:
        drop_cols.append(col)
X = df.drop(columns=drop_cols, errors="ignore")
y = df[TARGET].astype(str)

# Remove fully empty and constant columns; keep a stable feature set
X = X.dropna(axis=1, how="all")
constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
X = X.drop(columns=constant_cols, errors="ignore")
if X.shape[1] == 0:
    st.error("No usable feature columns were found after excluding the target and identifiers.")
    st.stop()
if y.value_counts().min() < 2:
    st.error("Each target class needs at least two rows to create a stratified train/test split.")
    st.stop()

numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
transformers = []
if numeric_features:
    transformers.append(("num", numeric_pipeline, numeric_features))
if categorical_features:
    transformers.append(("cat", categorical_pipeline, categorical_features))
preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

test_fraction = 0.2
class_counts = y.value_counts()
stratify = y if class_counts.min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_fraction, random_state=42, stratify=stratify
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=max(1, min(5, len(X_train)))),
    "Naive Bayes": GaussianNB(),
}

trained_models = {}
scores = {}
for name, classifier in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("classifier", classifier),
    ])
    try:
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)
        trained_models[name] = pipe
        scores[name] = accuracy_score(y_test, predictions)
    except Exception as exc:
        st.sidebar.warning(f"{name} could not be trained: {exc}")

if not trained_models:
    st.error("No model could be trained. Check your CSV data types and target values.")
    st.stop()

# ---- Navigation ----
page = st.sidebar.radio("Navigate", [
    "Executive Dashboard", "Loan Predictor", "Model Comparison", "Dataset Explorer"
])

# Common dashboard metrics
approval_label = "Approved"
approval_rate = (y == approval_label).mean() * 100 if approval_label in set(y) else np.nan
best_model_name = max(scores, key=scores.get)
best_score = scores[best_model_name]

if page == "Executive Dashboard":
    st.subheader("Executive Dashboard")
    st.caption("A quick overview of the dataset currently loaded in this session.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total applications", f"{len(df):,}")
    c2.metric("Input features", f"{X.shape[1]}")
    c3.metric("Approval share", f"{approval_rate:.1f}%" if not np.isnan(approval_rate) else "N/A")
    c4.metric("Best test accuracy", f"{best_score:.1%}", best_model_name)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Loan status distribution")
        status_counts = y.value_counts().rename_axis("Status").reset_index(name="Applications")
        st.bar_chart(status_counts.set_index("Status"))
    with right:
        st.markdown("#### Missing values by feature")
        missing = raw_df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0].head(12)
        if missing.empty:
            st.success("No missing values in the uploaded dataset.")
        else:
            st.bar_chart(missing.rename("Missing cells"))
    st.markdown("#### Dataset preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

elif page == "Loan Predictor":
    st.subheader("AI Loan Outcome Estimator")
    st.write("Enter applicant details below. The model estimates a pattern learned from the supplied dataset; it is not a lender decision.")
    selected_model = st.selectbox("Choose model", list(trained_models.keys()), index=list(trained_models.keys()).index(best_model_name))
    input_values = {}
    with st.form("loan_form"):
        columns = st.columns(2)
        for i, col in enumerate(X.columns):
            s = X[col]
            with columns[i % 2]:
                if col in numeric_features:
                    vals = pd.to_numeric(s, errors="coerce").dropna()
                    median = float(vals.median()) if not vals.empty else 0.0
                    low = float(vals.min()) if not vals.empty else 0.0
                    high = float(vals.max()) if not vals.empty else max(1.0, low + 1)
                    if low == high:
                        high = low + 1
                    # Use intuitive widgets for common binary fields
                    if col == "Credit_History":
                        input_values[col] = st.selectbox(col.replace("_", " "), [1.0, 0.0], index=0)
                    else:
                        input_values[col] = st.number_input(
                            col.replace("_", " "), min_value=low, max_value=high,
                            value=min(max(median, low), high), step=1.0 if (vals.dropna() % 1 == 0).all() else 0.1,
                            format="%.2f"
                        )
                else:
                    choices = s.dropna().astype(str).value_counts().index.tolist()
                    if not choices:
                        choices = ["Unknown"]
                    input_values[col] = st.selectbox(col.replace("_", " "), choices)
        submitted = st.form_submit_button("Predict loan outcome", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([input_values], columns=X.columns)
        try:
            prediction = trained_models[selected_model].predict(row)[0]
            st.divider()
            if str(prediction).lower() == "approved":
                st.success(f"Estimated outcome: **{prediction}**")
            else:
                st.warning(f"Estimated outcome: **{prediction}**")
            st.metric("Selected model test accuracy", f"{scores[selected_model]:.1%}")
            if hasattr(trained_models[selected_model], "predict_proba"):
                probabilities = trained_models[selected_model].predict_proba(row)[0]
                classes = trained_models[selected_model].named_steps["classifier"].classes_
                prob_df = pd.DataFrame({"Outcome": classes, "Model probability": probabilities})
                st.markdown("#### Model probability estimates")
                st.bar_chart(prob_df.set_index("Outcome"))
            st.caption("For educational demonstration only. Do not use this estimate as the sole basis for lending, credit, or financial decisions.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

elif page == "Model Comparison":
    st.subheader("Model Comparison")
    st.write("Models are trained on an 80% training split and evaluated on a 20% holdout split.")
    score_df = pd.DataFrame({
        "Model": list(scores.keys()),
        "Test accuracy": list(scores.values())
    }).sort_values("Test accuracy", ascending=False)
    score_df["Test accuracy (%)"] = (score_df["Test accuracy"] * 100).round(2)
    st.bar_chart(score_df.set_index("Model")["Test accuracy"])
    st.dataframe(score_df[["Model", "Test accuracy (%)"]], use_container_width=True, hide_index=True)
    st.info(f"Highest holdout accuracy in this run: **{best_model_name} ({best_score:.1%})**. Accuracy alone may hide class imbalance; evaluate precision, recall, F1-score and confusion matrix before drawing conclusions.")
    st.caption("Scores are recalculated from the dataset at app startup and are not a guarantee of future performance.")

elif page == "Dataset Explorer":
    st.subheader("Dataset Explorer")
    st.write("Filter, inspect and download the data currently loaded in the dashboard.")
    search = st.text_input("Search rows (text match across columns)")
    view = df.copy()
    if search.strip():
        mask = view.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        view = view[mask]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button(
        "Download displayed rows as CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="creditwise_dataset_view.csv",
        mime="text/csv",
    )
    st.markdown("#### Data quality")
    quality = pd.DataFrame({
        "Column": raw_df.columns,
        "Data type": [str(t) for t in raw_df.dtypes],
        "Missing values": [int(raw_df[c].isna().sum()) for c in raw_df.columns],
        "Unique values": [int(raw_df[c].nunique(dropna=True)) for c in raw_df.columns],
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)

st.divider()
st.markdown(
    "<p class='small-note'>CreditWise is an educational machine-learning demonstration. "
    "Predictions may be inaccurate and must not be used as the sole basis for real lending decisions.</p>",
    unsafe_allow_html=True,
)

