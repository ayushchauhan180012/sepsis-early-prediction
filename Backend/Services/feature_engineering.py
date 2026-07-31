import pandas as pd
from sqlalchemy import create_engine
import psycopg
from psycopg2 import sql

conn = psycopg.connect(
    host="",
    port=5432,
    dbname="prediction_cache",
    user="",
    password="2407"
)

cursor = conn.cursor()

engine = create_engine(
    "postgresql+psycopg2://username:password@localhost:5432/sepsis_db"
)

lab_cols = ['Lactate','WBC','Creatinine','Platelets']
vitals = ['HR','O2Sat','SBP','MAP','Resp','Temp']

def load_data(Patient_ID):
    Retrive_querry = sql.SQL("SELECT * FROM {}").format(sql.Identifier({Patient_ID}))
    df = pd.read_sql_query(Retrive_querry, engine)

    return df

# for Missing data handeling
def Missing_Handling(df: pd.DataFrame):

    # Missing indicators
    for col in lab_cols:
        df[col + '_missing'] = df[col].isnull().astype(int)

    # Forward fill vitals per patient
    df[vitals] = df.groupby('PatientID')[vitals].ffill()

    # Median fill 
    vital_medians = df[vitals].median()
    df[vitals] = df[vitals].fillna(vital_medians)
 


#  FEATURE ENGINEERING

def Add_Features(df: pd.DataFrame):

    baseline_window = 6
    # Delta6
    for col in vitals:
        
        # Delta1
        df[col + '_delta1'] = (
            df.groupby('PatientID')[col].shift(0) -
            df.groupby('PatientID')[col].shift(1)
        ).fillna(0)

        df[col + '_delta6'] = (
            df.groupby('PatientID')[col].shift(0) -
            df.groupby('PatientID')[col].shift(6)
        ).fillna(0)

            # Volatility (6-hour std)
        df[col + '_roll6_std'] = (
            df.groupby('PatientID')[col]
            .rolling(window=6, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            ).fillna(0)

        # Patient Baseline Deviation
        baseline_train = (
            df.groupby("PatientID")[col]
            .transform(lambda x: x.iloc[:baseline_window].mean())
        )
        
        df[col + "_baseline_dev"] = df[col] - baseline_train

        # Lab presence in last 6 hours
        df[col + "_recent_test"] = (
            df.groupby("PatientID")[col]
            .apply(lambda x: x.notnull().rolling(6, min_periods=1).max())
            .reset_index(level=0, drop=True)
        )

    # Clinical Ratio Features
    df["shock_index"] = df["HR"] / (df["SBP"] + 1)
    
    df["resp_o2_ratio"] = df["Resp"] / (df["O2Sat"] + 1)
    
    df["map_hr_ratio"] = df["MAP"] / (df["HR"] + 1)


    # Clinical Threshold Flags
    df["tachycardia"] = (df["HR"] > 100).astype(int)
    
    df["hypotension"] = (df["SBP"] < 90).astype(int)
    
    df["tachypnea"] = (df["Resp"] > 22).astype(int)

