import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import psutil
import os

# --- 1. SETUP & CACHING ---
st.set_page_config(
    page_title="AI Passport Module 4: Diabetes Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_clean_data():
    """Loads data and implements the notebook's imputation logic."""
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    
    # Notebook Logic: Replace 0s with median for continuous markers
    cols_to_fix = ['Insulin', 'BMI', 'Glucose', 'BloodPressure', 'SkinThickness']
    for col in cols_to_fix:
        median_val = df[col].median()
        df[col] = df[col].replace(0, median_val)
    return df

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.title("AI Passport Analysis")

# Dual-Track Selection Toggle
track = st.sidebar.radio(
    "Select Science Track:", 
    ["Clinical Science", "Foundational Science"],
    help="Toggle between a clinical healthcare focus or a foundational data science focus."
)

st.sidebar.markdown("---")

pages = [
    "Data Explorer",
    "Visual Analytics",
    "Model Training"
]

page = st.sidebar.radio(
    "Select Stage:", 
    pages,
    help="Navigate through the workflow from data inspection to model evaluation."
)

# --- SYSTEM MONITOR (CPU + RAM) ---
st.sidebar.markdown("---")
st.sidebar.subheader("System Monitor")
pid = os.getpid()
py = psutil.Process(pid)
memory_use = py.memory_info().rss / 1024 / 1024
cpu_use = psutil.cpu_percent(interval=1)

col1, col2 = st.sidebar.columns(2)
col1.metric("CPU Usage", f"{cpu_use}%")
col2.metric("RAM Usage", f"{memory_use:.0f} MB")
st.sidebar.caption("Real-time resource usage of this instance.")

# Initialize Data
df = load_and_clean_data()

# --- 3. PAGE LOGIC ---

# === PAGE: DATA EXPLORER ===
if page == "Data Explorer":
    if track == "Clinical Science":
        st.title("Patient Demographics & Vitals Explorer")
        st.markdown("""
        ### Instructions
        Use this page to review the patient cohort from the Pima Indians Diabetes dataset.
        1. **Clinical Data Handling**: In real-world clinical settings, missing lab results are common. Here, missing values (recorded as 0) for vitals like Insulin and BMI have been imputed with the cohort median to preserve patient records.
        2. **Patient Filtering**: Use the sliders below to subset the patient population by Age and BMI.
        """)
    else:
        st.title("Dataset & Feature Explorer")
        st.markdown("""
        ### Instructions
        Use this page to inspect the raw features of the diabetes dataset.
        1. **Data Imputation**: From a foundational data science perspective, 0 values in continuous biological variables represent missing data. The pipeline automatically imputes these with the feature median to prevent skewed model weights.
        2. **Feature Filtering**: Use the sliders below to subset the dataset based on variable ranges.
        """)

    with st.expander("Filter Options", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            age_range = st.slider("Select Age Range", 21, 81, (21, 81), help="Filter the table based on age.")
        with c2:
            bmi_range = st.slider("Select BMI Range", 18.0, 67.0, (18.0, 67.0), help="Filter the table based on BMI.")

    filtered_df = df[
        (df['Age'].between(age_range[0], age_range[1])) & 
        (df['BMI'].between(bmi_range[0], bmi_range[1]))
    ]

    st.subheader("Data Preview")
    st.dataframe(filtered_df, use_container_width=True)
    
    if track == "Clinical Science":
        st.caption(f"Showing {len(filtered_df)} patient records based on current filters.")
    else:
        st.caption(f"Showing {len(filtered_df)} samples based on current feature filters.")

# === PAGE: VISUAL ANALYTICS ===
elif page == "Visual Analytics":
    if track == "Clinical Science":
        st.title("Clinical Marker Analysis")
        st.markdown("""
        ### Instructions
        Visualizing clinical markers helps clinicians identify risk factors.
        1. **Marker Selection**: Choose a clinical vital from the dropdown.
        2. **Diagnostic Comparison**: The charts compare the distribution of the marker between healthy patients (0) and diabetic patients (1).
        """)
    else:
        st.title("Biomarker Distribution Analysis")
        st.markdown("""
        ### Instructions
        Visualizing data distributions is critical for understanding feature variance before training.
        1. **Feature Selection**: Choose a biomarker to visualize.
        2. **Class Separation**: Observe how well the feature separates the binary target classes (0 vs 1). Boxplots are especially useful for identifying outliers.
        """)
    
    col_feat, col_type = st.columns(2)
    with col_feat:
        feature = st.selectbox("Select Variable to Analyze", df.columns[:-1])
    with col_type:
        plot_type = st.radio("Plot Type", ["Distribution", "Boxplot"], horizontal=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    if plot_type == "Distribution":
        sns.histplot(data=df, x=feature, hue="Outcome", kde=True, ax=ax, palette="viridis")
        if track == "Clinical Science":
            ax.set_title(f"Patient Distribution of {feature} by Diagnosis")
        else:
            ax.set_title(f"Statistical Distribution of {feature} by Target Class")
    else:
        sns.boxplot(data=df, x="Outcome", y=feature, ax=ax, palette="magma")
        if track == "Clinical Science":
            ax.set_title(f"{feature} Spread between Healthy (0) and Diabetic (1) Patients")
        else:
            ax.set_title(f"Feature Variance and Outliers for {feature}")
    
    st.pyplot(fig)

# === PAGE: MODEL TRAINING ===
elif page == "Model Training":
    if track == "Clinical Science":
        st.title("Diagnostic Predictive Engine")
        st.markdown("""
        ### Instructions
        1. **Select Vitals**: Choose which patient vitals the diagnostic tool should use.
        2. **Choose Algorithm**: Logistic Regression is often preferred in clinical settings for its interpretability (odds ratios). SVM is used for complex, non-linear patient risk profiles.
        3. **Clinical Evaluation**: The ROC AUC score measures the tool's ability to correctly diagnose patients. The confusion matrix shows false positives vs. false negatives.
        """)
    else:
        st.title("Machine Learning Pipeline")
        st.markdown("""
        ### Instructions
        1. **Select Features**: Choose the independent variables for the predictive model.
        2. **Choose Algorithm**: Logistic Regression maps linear probabilities. SVM finds an optimal hyperplane in a higher-dimensional space and requires feature scaling (StandardScaler) to converge efficiently.
        3. **Model Evaluation**: The ROC AUC score is the primary metric for binary classification performance. The confusion matrix visualizes Type I and Type II errors.
        """)

    all_features = df.columns[:-1].tolist()
    selected_features = st.multiselect(
        "Select Predictors/Features:",
        all_features,
        default=['Glucose', 'BMI', 'Age', 'Insulin']
    )

    model_type = st.selectbox(
        "Choose Algorithm:",
        ["Logistic Regression", "Support Vector Machine (SVM)"]
    )

    if st.button("Train and Evaluate Model"):
        if not selected_features:
            st.error("Please select at least one variable.")
        else:
            X = df[selected_features]
            y = df['Outcome']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            with st.spinner("Processing..."):
                if model_type == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                else:
                    model = make_pipeline(StandardScaler(), SVC(probability=True, gamma='auto'))

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                # --- Results Display ---
                st.info(f"{model_type} training and evaluation complete.")
                
                m1, m2 = st.columns(2)
                m1.metric("ROC AUC Score", f"{roc_auc_score(y_test, y_prob):.3f}")
                
                with st.expander("View Full Classification Report"):
                    st.code(classification_report(y_test, y_pred))

                # Confusion Matrix Plot
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
                st.pyplot(fig)
