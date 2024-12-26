import os
import pandas as pd
import numpy as np

######################################
#            CONFIGURATION           #
######################################

# You can parameterize these if needed for Airflow:
INPUT_DATA_PATH = "/opt/airflow/data/diabetic_data.csv"
ADMISSION_TYPE_MAPPING = "/opt/airflow/data/admission_type_mapping.csv"
DISCHARGE_DISPOSITION_MAPPING = "/opt/airflow/data/discharge_disposition_mapping.csv"
ADMISSION_SOURCE_MAPPING = "/opt/airflow/data/admission_source_mapping.csv"

# S3 output path (bucket + key).
# Example: "s3://my-bucket/folder/diabetic_data_preprocessed.csv"
OUTPUT_CSV_S3_PATH = "s3://my-bucket/folder/diabetic_data_cleaned.csv"


######################################
#       LOW FREQUENCY MERGER         #
######################################

def merge_low_freq_categories(df, column, threshold=100):
    """
    Merge categories in the specified column that occur less than 'threshold' times into 'Other'.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the column to modify.
    column : str
        The column name whose categories you want to merge.
    threshold : int, optional (default=100)
        Categories with frequency below this threshold will be merged into 'Other'.

    Returns:
    --------
    df : pd.DataFrame
        The modified dataframe with low-frequency categories merged.
    """
    freq = df[column].value_counts()
    low_freq_cats = freq[freq < threshold].index
    df[column] = df[column].replace(low_freq_cats, 'Other')
    return df


######################################
#         ICD MAPPING FUNCTION       #
######################################

def map_icd_to_category(icd):
    """
    Converts ICD-9 codes into broad disease categories. 
    For simplicity, numeric ranges are used to approximate classification.
    """
    if icd == 'Unknown':
        return 'Unknown'
    icd = str(icd)
    try:
        code_num = float(icd)
        if 1 <= code_num < 140:
            return 'Infectious_and_Parasitic'
        elif 140 <= code_num < 240:
            return 'Neoplasms'
        elif 240 <= code_num < 280:
            return 'Endocrine_Metabolic'
        elif 280 <= code_num < 290:
            return 'Blood_And_Blood_forming_Organs'
        else:
            return 'Other'
    except ValueError:
        if icd.startswith('V'):
            return 'Supplementary_Factors'
        elif icd.startswith('E'):
            return 'External_Causes_of_Injury'
        else:
            return 'Other'


######################################
#        MAIN PREPROCESSING          #
######################################

def preprocess_data():
    # 1. Read raw data
    df = pd.read_csv(INPUT_DATA_PATH)

    # 2. Normalize 'max_glu_serum' and 'A1Cresult'
    df['max_glu_serum'] = df['max_glu_serum'].replace({'Norm': 'normal'}).fillna('none')
    df['A1Cresult'] = df['A1Cresult'].replace({'Norm': 'normal'}).fillna('none')

    # 3. Drop 'weight' due to excessive missingness
    df.drop(columns=['weight'], inplace=True)

    # 4. Replace missing or '?' with 'Unknown' in payer_code and medical_specialty
    for col in ['payer_code', 'medical_specialty']:
        df[col] = df[col].fillna('Unknown').replace('?', 'Unknown')

    # 5. Convert diag_1, diag_2, diag_3 to broader ICD categories
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col] = df[col].fillna('Unknown').apply(map_icd_to_category)

    # 6. Read and map admission_type, discharge_disposition, admission_source
    admission_type_map = pd.read_csv(ADMISSION_TYPE_MAPPING, usecols=['admission_type_id','description']).dropna()
    discharge_disposition_map = pd.read_csv(DISCHARGE_DISPOSITION_MAPPING, usecols=['discharge_disposition_id','description']).dropna()
    admission_source_map = pd.read_csv(ADMISSION_SOURCE_MAPPING, usecols=['admission_source_id','description']).dropna()

    admission_type_dict = dict(zip(admission_type_map['admission_type_id'], admission_type_map['description']))
    discharge_disposition_dict = dict(zip(discharge_disposition_map['discharge_disposition_id'], discharge_disposition_map['description']))
    admission_source_dict = dict(zip(admission_source_map['admission_source_id'], admission_source_map['description']))

    df['admission_type'] = df['admission_type_id'].map(admission_type_dict)
    df['discharge_disposition'] = df['discharge_disposition_id'].map(discharge_disposition_dict)
    df['admission_source'] = df['admission_source_id'].map(admission_source_dict)

    # Drop the original ID columns
    df.drop(columns=['admission_type_id','discharge_disposition_id','admission_source_id'], inplace=True)

    # Standardize unknown categories
    for col in ['admission_type', 'discharge_disposition', 'admission_source']:
        df[col] = df[col].replace(['Not Mapped', 'NULL', 'Unknown/Invalid'], 'Unknown').fillna('Unknown')

    # 7. Combine >30 and NO into "Not <30"
    df['readmitted_binary'] = df['readmitted'].apply(lambda x: '<30' if x == '<30' else 'Not <30')

    # 8. Merge low-frequency categories in selected columns
    for col in ['medical_specialty', 'payer_code', 'admission_type', 'discharge_disposition', 'admission_source']:
        df = merge_low_freq_categories(df, col, threshold=100)

    # 9. Write the preprocessed DataFrame as a CSV to S3 (requires `s3fs` and IAM permissions)
    df.to_csv(OUTPUT_CSV_S3_PATH, index=False)
    print(f"Successfully wrote preprocessed data to {OUTPUT_CSV_S3_PATH}")


######################################
#       ENTRY POINT (if needed)      #
######################################

if __name__ == "__main__":
    preprocess_data()