# data_cleaning.py

import pandas as pd
import numpy as np

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the column names of a DataFrame.
    """
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df = df.rename(columns={'st': 'state', 'income': 'customer_income'})
    return df

def standardize_gender(df):
    if 'gender' in df.columns:
        # Garantir que estamos a trabalhar com uma Series
        gender_series = df['gender']

        # Converter para string e remover nulos
        gender_series = gender_series.astype(str).fillna('')

        # Padronizar valores
        df['gender'] = gender_series.apply(lambda x: 'F' if x.strip().upper().startswith('F') else
                                                      'M' if x.strip().upper().startswith('M') else x)
    return df


def standardize_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the 'state' column to full state names.
    """
    if 'state' in df.columns:
        state_mapping = {
            'AL': 'ALABAMA', 'AK': 'ALASKA', 'AZ': 'ARIZONA', 'AR': 'ARKANSAS', 'CA': 'CALIFORNIA', 'CALI': 'CALIFORNIA',
            'CO': 'COLORADO', 'CT': 'CONNECTICUT', 'DE': 'DELAWARE', 'FL': 'FLORIDA', 'GA': 'GEORGIA',
            'HI': 'HAWAII', 'ID': 'IDAHO', 'IL': 'ILLINOIS', 'IN': 'INDIANA', 'IA': 'IOWA',
            'KS': 'KANSAS', 'KY': 'KENTUCKY', 'LA': 'LOUISIANA', 'ME': 'MAINE', 'MD': 'MARYLAND',
            'MA': 'MASSACHUSETTS', 'MI': 'MICHIGAN', 'MN': 'MINNESOTA', 'MS': 'MISSISSIPPI', 'MO': 'MISSOURI',
            'MT': 'MONTANA', 'NE': 'NEBRASKA', 'NV': 'NEVADA', 'NH': 'NEW HAMPSHIRE', 'NJ': 'NEW JERSEY',
            'NM': 'NEW MEXICO', 'NY': 'NEW YORK', 'NC': 'NORTH CAROLINA', 'ND': 'NORTH DAKOTA', 'OH': 'OHIO',
            'OK': 'OKLAHOMA', 'OR': 'OREGON', 'PA': 'PENNSYLVANIA', 'RI': 'RHODE ISLAND', 'SC': 'SOUTH CAROLINA',
            'SD': 'SOUTH DAKOTA', 'TN': 'TENNESSEE', 'TX': 'TEXAS', 'UT': 'UTAH', 'VT': 'VERMONT',
            'VA': 'VIRGINIA', 'WA': 'WASHINGTON', 'WV': 'WEST VIRGINIA', 'WI': 'WISCONSIN', 'WY': 'WYOMING'
        }
        df['state'] = df['state'].astype(str).str.upper().replace(state_mapping)
    return df

def standardize_education(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the 'education' column.
    """
    if 'education' in df.columns:
        df['education'] = df['education'].astype(str)
        df.loc[df['education'].str.contains(r'^[Bb]', na=False), 'education'] = 'Bachelor'
    return df

def standardize_vehicle_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the 'vehicle_class' column by grouping similar values.
    """
    if 'vehicle_class' in df.columns:
        df['vehicle_class'] = df['vehicle_class'].astype(str)
        df.loc[df['vehicle_class'].str.contains(r'^[Lu]', na=False), 'vehicle_class'] = 'Luxury'
        df.loc[df['vehicle_class'].str.contains(r'\bSports\b', na=False), 'vehicle_class'] = 'Luxury'
    return df

def clean_and_convert_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and converts numerical columns.
    """
    if 'customer_lifetime_value' in df.columns:
        df['customer_lifetime_value'] = df['customer_lifetime_value'].astype(str).str.rstrip('%')
        df['customer_lifetime_value'] = pd.to_numeric(df['customer_lifetime_value'], errors='coerce').astype('float64')

    if 'number_of_open_complaints' in df.columns:
        df['number_of_open_complaints'] = df['number_of_open_complaints'].astype(str).str.extract(r'/(\d+)/')
        df['number_of_open_complaints'] = pd.to_numeric(df['number_of_open_complaints'], errors='coerce').astype('Int64')
    
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows from the DataFrame, keeping only the first occurrence.
    """
    initial_rows = len(df)
    df_cleaned = df.drop_duplicates(keep='first').reset_index(drop=True)
    rows_removed = initial_rows - len(df_cleaned)
    print(f"Number of duplicate rows removed: {rows_removed}")
    return df_cleaned

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills in missing values in both categorical and numerical columns.
    """
    categorical_cols = ['customer', 'state', 'gender', 'education', 'policy_type', 'vehicle_class']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    numerical_cols = ['total_claim_amount', 'monthly_premium_auto', 'customer_income', 'customer_lifetime_value']
    for col in numerical_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
    
    if 'number_of_open_complaints' in df.columns:
        df['complaints_missing'] = df['number_of_open_complaints'].isnull()
    
    return df

def main_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to execute the complete data cleaning and formatting pipeline.
    """
    print("Starting the data cleaning and formatting pipeline...")
    
    df = clean_column_names(df)
    df = standardize_gender(df)
    df = standardize_state(df)
    df = standardize_education(df)
    df = standardize_vehicle_class(df)
    df = clean_and_convert_numerical(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df) 
    
    print("Data cleaning pipeline completed successfully.")
    return df