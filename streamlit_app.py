import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import psutil
import os

# --- 1. SETUP & CACHING ---
st.set_page_config(
    page_title="Evaluating Machine Learning Models",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def build_eicu_data():
    """Loads and merges the raw eICU data."""
    # Load patient information
    patient_cols = ['patientunitstayid', 'hospitalid', 'gender', 'age', 'ethnicity', 'admissionheight', 'admissionweight', 'dischargeweight',
                'hospitaladmitsource', 'hospitaldischargelocation', 'hospitaldischargestatus', 'unittype', 'uniquepid', 'unitvisitnumber',
                'patienthealthsystemstayid', 'hospitaldischargeyear']
    df = pd.read_csv('https://www.dropbox.com/scl/fi/qld4pvo6vlptm41av3y2e/patient.csv?rlkey=gry21fvb3u3dytz7i5jcujcu9&dl=1',
                      usecols=patient_cols)
    
    # Clean Age for numerical filtering
    df['age'] = df['age'].replace({'> 89': 90})
    df['age'] = pd.to_numeric(df['age'], errors='coerce') 
    
    df = df.sort_values(by=['uniquepid', 'patienthealthsystemstayid', 'unitvisitnumber'])
    df = df.groupby('patienthealthsystemstayid').first().reset_index()
    df = df.drop(columns=['unitvisitnumber'])

    # Load hospital information
    hospital = pd.read_csv('https://www.dropbox.com/scl/fi/5sdjsbrjxk0hlbmpb4csi/hospital.csv?rlkey=rmlrhg3m9sm3hj2s6rykrg5w2&st=329j4gwk&dl=1')
    with pd.option_context('future.no_silent_downcasting', True):
        hospital = hospital.replace({'teachingstatus': {'f': 0, 't':1}})

    df = df.merge(hospital, on='hospitalid', how='left')

    # Load labs
    labcols = ['patientunitstayid', 'labname', 'labresult']
    
    # Updated alphabetized list to match pivot_table output
    labnames = ['BUN', 'Hct', 'Hgb', 'MCH', 'MCHC', 'MCV', 'RBC', 'RDW', 'albumin', 'bicarbonate', 'calcium', 'chloride', 'creatinine', 'glucose', 'platelets', 'potassium', 'sodium', 'wbc']
    
    labs = pd.read_csv('https://www.dropbox.com/scl/fi/qaxtx330hicc5u61siehn/lab.csv?rlkey=xs9oxpl5istkbuh5s80oyxwwi&st=ydfrjxkh&dl=1',
                        usecols=labcols)
    labs['labname'] = labs['labname'].replace({
        'WBC x 1000': 'wbc',
        'platelets x 1000': 'platelets'
    })
    labs = labs[labcols]
    labs = labs[labs['labname'].isin(labnames)]
    labs = labs.pivot_table(columns=['labname'], values=['labresult'], aggfunc='mean', index='patientunitstayid')
    labs.columns = list(labs.columns.droplevel(0))
    labs = labs.reset_index()
    
    # Apply the 'lab_' prefix
    labnames = ['lab_' + c.lower() for c in labnames]
    labs.columns = ['patientunitstayid'] + labnames

    df = df.merge(labs, on='patientunitstayid', how='left')

    # Renaming
    df = df.rename(columns={
        'uniquepid': 'patient_id',
        'patienthealthsystemstayid': 'admission_id',
        'hospitaldischargeyear': 'admission_year',
        'hospitalid': 'hospital_id',
        'admissionheight': 'height',
        'admissionweight': 'weight_admission',
        'dischargeweight': 'weight_discharge',
        'region': 'hospital_region',
        'teachingstatus': 'hospital_teaching',
        'numbedscategory': 'hospital_beds',
        'hospitaladmitsource': 'admission_source',
        'hospitaldischargelocation': 'discharge_location'
    })

    df['in_hospital_mortality'] = df['hospitaldischargestatus'].map(lambda status: {'Alive': 0, 'Expired': 1}[status]
                                                                    if pd.notnull(status) else status)
    df = df.drop(columns=['hospitaldischargestatus'])

    df_cols = ['patient_id', 'admission_id', 'admission_year', 'age', 'gender', 'ethnicity', 'height', 'weight_admission',
               'weight_discharge', 'admission_source', 'hospital_id', 'hospital_region', 'hospital_teaching',
               'hospital_beds'] + labnames + ['discharge_location', 'in_hospital_mortality']
    df = df[df_cols]
    df = df.sort_values(by=['admission_year', 'patient_id', 'admission_id'])
    df = df.reset_index(drop=True)
    return df

@st.cache_data
def get_processed_data(df_raw):
    """Runs the preprocessing steps (One-Hot Encoding & Imputation)."""
    # Drop IDs for processing
    x = df_raw.drop(columns=['patient_id', 'hospital_id', 'admission_id', 'admission_year', 'weight_discharge', 'discharge_location'])
    
    numerical_features = ['age', 'height','weight_admission', 'hospital_teaching', 
                          'lab_bun', 'lab_hct', 'lab_hgb', 'lab_mch', 'lab_mchc', 
                          'lab_mcv', 'lab_rbc', 'lab_rdw', 'lab_albumin', 'lab_bicarbonate', 
                          'lab_calcium', 'lab_chloride', 'lab_creatinine', 'lab_glucose', 
                          'lab_platelets', 'lab_potassium', 'lab_sodium', 'lab_wbc']

    categorical_features = ['ethnicity', 'admission_source', 'hospital_region', 'hospital_beds']
    
    x[numerical_features] = x[numerical_features].apply(pd.to_numeric, errors='coerce', axis=1)
    x[numerical_features] = x[numerical_features].fillna(x[numerical_features].mean())
    
    # Standard preprocessing
    x = pd.get_dummies(x, columns=categorical_features, dtype='int')
    x['in_hospital_mortality'] = x['in_hospital_mortality'].fillna(0)
    
    return x

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.title("Evaluating Machine Learning Models")

track = st.sidebar.radio(
    "Select Science Track:", 
    ["Clinical Science", "Foundational Science"],
    help="Toggle between a clinical healthcare focus or a foundational data science focus."
)

st.sidebar.markdown("---")

# Define pages
pages = ["1. Data Processing", "2. Exploratory Analysis", "3. Univariate Analysis", "4. Multivariate Analysis"]

# Main Menu
page = st.sidebar.radio(
    "Select Stage:", 
    pages,
    help="Navigate through the 4 steps of the analysis pipeline."
)

# --- SYSTEM MONITOR (Bottom of Sidebar) ---
st.sidebar.markdown("---")
st.sidebar.subheader("System Monitor")
pid = os.getpid()
py = psutil.Process(pid)
memory_use = py.memory_info().rss / 1024 / 1024  # Memory in MB
cpu_use = psutil.cpu_percent(interval=None)     # CPU Percent (non-blocking)

col1, col2 = st.sidebar.columns(2)
col1.metric("CPU", f"{cpu_use}%")
col2.metric("RAM", f"{memory_use:.0f} MB")
st.sidebar.caption("Real-time resource usage of this app instance.")

# --- 3. PAGE LOGIC ---

# === 1. DATA PROCESSING ===
if page == "1. Data Processing":
    if track == "Clinical Science":
        st.title("Clinical Data Processing")
        st.markdown("""
        ### Preparing Electronic Health Records
        In clinical environments, raw patient records often contain missing vitals and unique hospital identifiers that offer no predictive value.
        
        ### Pipeline Operations
        1. **Remove Identifiers**: Exclude random numeric identifiers (e.g., patient IDs) to maintain privacy and prevent model bias.
        2. **Imputation**: Handle missing clinical lab results by imputing the cohort mean to preserve the patient record for analysis.
        """)
    else:
        st.title("Data Cleaning Engine")
        st.markdown("""
        ### Dataset Feature Engineering
        Before training, raw datasets must be standardized. Variables containing no informational value or target leakage (like discharge weight) are removed.
        
        ### Pipeline Operations
        1. **One-Hot Encoding**: Transform categorical string variables into binary matrices.
        2. **Mean Imputation**: Fill missing numerical values with the column mean to ensure matrices remain dense for mathematical convergence.
        """)

    # Initialize state
    if 'proc_run' not in st.session_state:
        st.session_state.proc_run = False
        
    with st.spinner("Loading Raw Data..."):
        df_raw = build_eicu_data()

    if st.button("Run Preprocessing Pipeline", help="Clean the data, impute missing values, and prepare it for modeling."):
        st.session_state.proc_run = True

    if st.session_state.proc_run:
        with st.spinner("Processing..."):
            df_processed = get_processed_data(df_raw)
            
        st.success("Preprocessing Complete")
        st.write(f"Processed Data Shape: {df_processed.shape}")
        
        st.subheader("Processed Data Preview")
        st.dataframe(df_processed.head(), use_container_width=True)
        
        if track == "Clinical Science":
            st.caption("A preview of the first 5 rows of the clean patient cohort.")
        else:
            st.caption("A preview of the first 5 samples of the encoded dataset.")

# === 2. EXPLORATORY ANALYSIS ===
elif page == "2. Exploratory Analysis":
    if track == "Clinical Science":
        st.title("Patient Demographics & Biomarkers")
        st.markdown("Use this explorer to visualize clinical markers and demographic breakdowns across the hospital network.")
    else:
        st.title("Feature Variance & Distributions")
        st.markdown("Use this explorer to visualize feature variance and observe data distributions prior to modeling.")
    
    with st.spinner("Loading Data Explorer..."):
        df_raw = build_eicu_data()
        df_explorer = df_raw.drop(columns=['patient_id', 'hospital_id', 'admission_id', 'admission_year', 'weight_discharge', 'discharge_location'])
        
        df_processed = get_processed_data(df_raw)
        if 'gender' not in df_processed.columns:
            df_processed['gender'] = df_raw['gender']

    # --- DATA EXPLORER FILTERS ---
    with st.expander("Filter Options", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            all_regions = df_explorer['hospital_region'].dropna().unique().tolist()
            selected_regions = st.multiselect("Hospital Region:", all_regions, default=all_regions)
        with c2:
            all_genders = df_explorer['gender'].dropna().unique().tolist()
            selected_genders = st.multiselect("Gender:", all_genders, default=all_genders)
            
        st.write("---")
        c3, c4, c5 = st.columns(3)
        with c3:
            min_age, max_age = int(df_explorer['age'].min()), int(df_explorer['age'].max())
            age_range = st.slider("Patient Age:", min_age, max_age, (min_age, max_age))
        with c4:
            min_h = float(df_explorer['height'].dropna().min())
            max_h = float(df_explorer['height'].dropna().max())
            height_range = st.slider("Height (cm):", min_h, max_h, (min_h, max_h))
        with c5:
            min_w = float(df_explorer['weight_admission'].dropna().min())
            max_w = float(df_explorer['weight_admission'].dropna().max())
            weight_range = st.slider("Weight (kg):", min_w, max_w, (min_w, max_w))

    # Apply Filters
    mask = (
        (df_explorer['hospital_region'].isin(selected_regions)) &
        (df_explorer['gender'].isin(selected_genders)) &
        (df_explorer['age'].between(age_range[0], age_range[1])) &
        (df_explorer['height'].between(height_range[0], height_range[1])) &
        (df_explorer['weight_admission'].between(weight_range[0], weight_range[1]))
    )
    df_filtered = df_explorer[mask]

    st.dataframe(df_filtered, height=300, use_container_width=True)
    st.divider()

    # --- VISUALIZATION ---
    if track == "Clinical Science":
        st.subheader("Sex-Specific Clinical Patterns")
        st.markdown("Do male and female patients exhibit different baseline physiological patterns? In clinical research, aggregating all patients can mask critical biological differences.")
    else:
        st.subheader("Class Separation by Subgroup")
        st.markdown("Do features distribute differently depending on subgroup categorizations? Pooling disparate subgroups can hide variance and confound model weights.")

    available_vars = ['age', 'height','weight_admission', 'lab_bun', 'lab_hct', 'lab_hgb', 'lab_mch', 'lab_mchc', 
                      'lab_mcv', 'lab_rbc', 'lab_rdw', 'lab_albumin', 'lab_bicarbonate', 
                      'lab_calcium', 'lab_chloride', 'lab_creatinine', 'lab_glucose', 
                      'lab_platelets', 'lab_potassium', 'lab_sodium', 'lab_wbc']
    
    selected_variables = st.multiselect(
        "Select Variables to Compare:", 
        available_vars, 
        default=['lab_bun','lab_creatinine','lab_sodium']
    )

    if selected_variables:
        df_male = df_processed[df_processed['gender'] == "Male"]
        df_female = df_processed[df_processed['gender'] == "Female"]

        female_survived = [df_female.loc[df_female['in_hospital_mortality'] == 0][str(i)].mean() for i in selected_variables]
        female_dead = [df_female.loc[df_female['in_hospital_mortality'] == 1][str(i)].mean() for i in selected_variables]
        male_survived = [df_male.loc[df_male['in_hospital_mortality'] == 0][str(i)].mean() for i in selected_variables]
        male_dead = [df_male.loc[df_male['in_hospital_mortality'] == 1][str(i)].mean() for i in selected_variables]

        Year = ['Survival'] * len(female_survived) + ['In hospital mortality'] * len(female_dead)
        Female = female_survived + female_dead
        Male = male_survived + male_dead
        
        df_plot = pd.DataFrame({'index': selected_variables * 2, 'Year': Year, 'Female': Female, 'Male': Male})
        df_plot.set_index(['Year', 'index'], inplace=True)
        df0 = df_plot.reorder_levels(['index', 'Year']).sort_index().unstack(level=-1)

        colors = plt.cm.Paired.colors
        fig, ax = plt.subplots(figsize=(10, 6))
        (df0['Female'] + df0['Male']).plot(kind='barh', color=[colors[3], colors[2]], rot=0, ax=ax)
        df0['Male'].plot(kind='barh', color=[colors[5], colors[4]], rot=0, ax=ax)
        ax.legend([f'{val} ({context})' for val, context in df0.columns])
        ax.set_title("Mean Values by Mortality Status (Stacked by Subgroup)")
        st.pyplot(fig)
        
        with st.expander("View Chart Data as Table"):
            st.dataframe(df0, use_container_width=True)

# === 3. UNIVARIATE ANALYSIS ===
elif page == "3. Univariate Analysis":
    if track == "Clinical Science":
        st.title("Sex-Specific Clinical Risk Factors")
        st.markdown("We will perform univariate analyses using odds ratios. This helps uncover whether specific clinical labs hold differential predictive risk for mortality in women versus men.")
    else:
        st.title("Univariate Feature Importance")
        st.markdown("We will perform univariate regression to calculate odds ratios. This establishes the baseline independent predictive power of each feature before complex modeling.")

    with st.spinner("Preparing Data & Models..."):
        df_raw = build_eicu_data()
        df_processed = get_processed_data(df_raw)
        if 'gender' not in df_processed.columns:
            df_processed['gender'] = df_raw['gender']

        df_male = df_processed[df_processed['gender'] == "Male"]
        df_female = df_processed[df_processed['gender'] == "Female"]

        train_dat_female, test_dat_female = train_test_split(df_female, test_size=0.2, random_state=2025)
        train_dat_male, test_dat_male = train_test_split(df_male, test_size=0.2, random_state=2025)
        train_dat_full, test_dat_full = train_test_split(df_processed, test_size=0.2, random_state=2025)

    all_vars = ['age', 'height','weight_admission', 'lab_bun', 'lab_hct', 'lab_hgb', 'lab_mch', 'lab_mchc', 
                      'lab_mcv', 'lab_rbc', 'lab_rdw', 'lab_albumin', 'lab_bicarbonate', 
                      'lab_calcium', 'lab_chloride', 'lab_creatinine', 'lab_glucose', 
                      'lab_platelets', 'lab_potassium', 'lab_sodium', 'lab_wbc']
    
    st.subheader("Calculate Odds Ratios")
    selected_variables = st.multiselect("Select Variables:", all_vars, default=all_vars[:10])

    if st.button("Run Regression Models"):
        vals_male, vals_female, vals_full = [], [], []
        
        progress_bar = st.progress(0)
        for i, univ_analysis in enumerate(selected_variables):
            progress_bar.progress((i + 1) / len(selected_variables))
            
            # Male
            reg_male = smf.logit(f'in_hospital_mortality ~ {univ_analysis}', data=train_dat_male).fit(disp=0)
            vals_male.append(np.exp(reg_male.params).iloc[1])
            
            # Female
            reg_female = smf.logit(f'in_hospital_mortality ~ {univ_analysis}', data=train_dat_female).fit(disp=0)
            vals_female.append(np.exp(reg_female.params).iloc[1])
            
            # Full
            reg_full = smf.logit(f'in_hospital_mortality ~ {univ_analysis}', data=train_dat_full).fit(disp=0)
            vals_full.append(np.exp(reg_full.params).iloc[1])

        progress_bar.empty()

        fig, ax = plt.subplots(figsize=(12, len(selected_variables) * 0.6 + 2))
        bar_width = 0.25
        r1 = np.arange(len(selected_variables))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        ax.barh(r1, vals_female, bar_width, label='Female', color='tab:orange', alpha=0.8)
        ax.barh(r2, vals_male, bar_width, label='Male', color='tab:blue', alpha=0.8)
        ax.barh(r3, vals_full, bar_width, label='All Cohort', color='tab:green', alpha=0.8)
        
        ax.set_xlabel('Odds Ratio (OR)')
        ax.set_yticks([r + bar_width for r in range(len(selected_variables))])
        ax.set_yticklabels(selected_variables)
        ax.set_title('Univariate Analysis: Odds Ratios')
        ax.legend()
        ax.axvline(x=1, color='gray', linestyle='--', linewidth=0.8)
        st.pyplot(fig)
        
        with st.expander("View OR Data as Table"):
            df_or = pd.DataFrame({
                "Variable": selected_variables,
                "OR (Female)": vals_female,
                "OR (Male)": vals_male,
                "OR (All)": vals_full
            })
            st.dataframe(df_or, use_container_width=True)

# === 4. MULTIVARIATE ANALYSIS ===
elif page == "4. Multivariate Analysis":
    if track == "Clinical Science":
        st.title("Diagnostic Predictive Engine")
        st.markdown("We will evaluate how well our multivariate models perform within each sex subgroup using the AUROC metric for diagnostic discrimination.")
    else:
        st.title("Machine Learning Pipeline Evaluation")
        st.markdown("We will measure the discriminative ability of our multivariate classifiers across specific demographic splits using the AUROC metric.")

    with st.spinner("Preparing Data..."):
        df_raw = build_eicu_data()
        df_processed = get_processed_data(df_raw)
        if 'gender' not in df_processed.columns:
            df_processed['gender'] = df_raw['gender']

    st.markdown("#### 1. Population Filter (Optional)")
    enable_filter = st.checkbox("Filter Dataset?")
    df_model_input = df_processed.copy()
    
    if enable_filter:
        numeric_options = ['age', 'weight_admission', 'height', 'lab_creatinine']
        filter_col = st.selectbox("Select Feature:", numeric_options)
        min_val, max_val = float(df_processed[filter_col].min()), float(df_processed[filter_col].max())
        range_vals = st.slider(f"Range for {filter_col}:", min_val, max_val, (min_val, max_val))
        df_model_input = df_processed[(df_processed[filter_col] >= range_vals[0]) & (df_processed[filter_col] <= range_vals[1])]

    st.markdown("#### 2. Model Features")
    all_vars = ['age', 'height','weight_admission', 'lab_bun', 'lab_hct', 'lab_hgb', 'lab_mch', 'lab_mchc', 
                      'lab_mcv', 'lab_rbc', 'lab_rdw', 'lab_albumin', 'lab_bicarbonate', 
                      'lab_calcium', 'lab_chloride', 'lab_creatinine', 'lab_glucose', 
                      'lab_platelets', 'lab_potassium', 'lab_sodium', 'lab_wbc']
    selected_predictors = st.multiselect("Choose predictors:", all_vars, default=all_vars)

    if st.button("Train & Evaluate Models"):
        if not selected_predictors:
            st.error("Select at least one predictor.")
            st.stop()
            
        df_male = df_model_input[df_model_input['gender'] == "Male"]
        df_female = df_model_input[df_model_input['gender'] == "Female"]
        
        train_dat_female, test_dat_female = train_test_split(df_female, test_size=0.2, random_state=2025)
        train_dat_male, test_dat_male = train_test_split(df_male, test_size=0.2, random_state=2025)
        train_dat, test_dat = train_test_split(df_model_input, test_size=0.2, random_state=2025)

        vals_male, vals_female, vals_full = [], [], []

        try:
            # Male
            reg_male = smf.logit(f"in_hospital_mortality ~ " + " + ".join(selected_predictors), data=train_dat_male).fit(disp=0)
            vals_male.append(roc_auc_score(test_dat_male['in_hospital_mortality'], reg_male.predict(test_dat_male[selected_predictors])))
            
            # Female
            reg_female = smf.logit(f"in_hospital_mortality ~ " + " + ".join(selected_predictors), data=train_dat_female).fit(disp=0)
            vals_female.append(roc_auc_score(test_dat_female['in_hospital_mortality'], reg_female.predict(test_dat_female[selected_predictors])))
            
            # Full
            reg = smf.logit(f"in_hospital_mortality ~ " + " + ".join(selected_predictors), data=train_dat).fit(disp=0)
            vals_full.append(roc_auc_score(test_dat['in_hospital_mortality'], reg.predict(test_dat[selected_predictors])))
            
            
            fig, ax = plt.subplots(figsize=(8, 5))
            r = np.arange(1)
            width = 0.25
            
            plt.bar(r, vals_female, width, label='Female', color='tab:orange')
            plt.bar(r + width, vals_male, width, label='Male', color='tab:blue')
            plt.bar(r + 2*width, vals_full, width, label='All Cohort', color='tab:green')
            
            plt.ylabel('AUROC')
            plt.title('Multivariate Analysis Performance')
            plt.xticks([])
            plt.ylim(0.5, 1.0)
            plt.legend(loc='lower right')
            
            # Add labels
            plt.text(r - 0.05, vals_female[0] + 0.01, f"{vals_female[0]:.3f}")
            plt.text(r + width - 0.05, vals_male[0] + 0.01, f"{vals_male[0]:.3f}")
            plt.text(r + 2*width - 0.05, vals_full[0] + 0.01, f"{vals_full[0]:.3f}")
            
            st.pyplot(fig)
            
            with st.expander("View Results as Table"):
                res_df = pd.DataFrame({
                    "Group": ["Female", "Male", "All Cohort"],
                    "AUROC": [vals_female[0], vals_male[0], vals_full[0]]
                })
                st.dataframe(res_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error training model: {e}")
