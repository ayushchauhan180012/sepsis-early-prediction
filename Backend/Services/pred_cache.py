from Database.querry import check_Patient_exists,Create_Patient, Patient_Cache_Manage , Update_Patient_Cache

from feature_engineering import (
    load_data,
    Missing_Handling,
    Add_Features
)


def Prediction_Pipeline(patient_id: str, patient_details: dict, values: list):


    # Create patient if first visit
    if not check_Patient_exists(patient_id):
        Create_Patient(patient_id, patient_details)


    # Store latest ICU record
    Patient_Cache_Manage(patient_id, values)


    # Load last 6 rows
    df = load_data(patient_id)

    # Missing value handling
    Missing_Handling(df)


    # Feature Engineering
    Add_Features(df)

    # Update database with new features
    columns = df.columns

    set_clause = ", ".join(f"{col}=%s" for col in columns)

    update_values = [df.iloc[-1][c] for c in columns]
    update_values.append(df.iloc[-1]["ICULOS"])

    Update_Patient_Cache(
        patient_id,
        update_values,
        set_clause
    )
    return df.tail(1)


