from sqlalchemy import create_engine
import psycopg
from psycopg2 import sql
from feature_engineering import load_data, Add_Features ,Missing_Handling


engine = create_engine(
    "postgresql+psycopg2://username:password@localhost:5432/sepsis_db"
)

conn = psycopg.connect(
    host="",
    port=5432,
    dbname="postgres",
    user="",
    password="2407"
)

cursor = conn.cursor()

# check of database exist 
def check_db_exixt():  # this will be in sim_data.py
    cursor.execute(""" SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = predicion_cache);""",)

    if (cursor.fetchone()[0]==False):
        cursor.execute(""" CREATE DATABASE prediction_cache;""")
        cursor.close()
        conn.close()
        conn = psycopg.connect(
            host="",
            port=5432,
            dbname="prediction_cache",
            user="",
            password="2407"
        )
        cursor = conn.cursor()

def create_patient(Patient_ID): # this will be in sim_data.py

    cursor.execute("""SELECT EXISTS ( SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s);""",(Patient_ID,))

    if (cursor.fetchone()[0]==False):
        cursor.execute("""CREATE TABLE %s(
            HR FLOAT, O2Sat FLOAT, SBP FLOAT, MAP FLOAT, Resp FLOAT, Temp FLOAT ,Lactate FLOAT ,WBC FLOAT, Creatinine FLOAT, Platelets FLOAT, Age INT, ICULOS INT NOT NULL);""", (Patient_ID,))

def data_IO_manage(Patient_ID, values): # this will be in sim_data.py

    cursor.execute(
        sql.SQL("""
            INSERT INTO {} (HR, O2Sat, SBP, MAP, Resp, Temp, Lactate, WBC, Creatinine, Platelets, Age, ICULOS)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);""").format(sql.Identifier(Patient_ID)),values)

    cursor.execute(
        sql.SQL("""
            DELETE FROM {} WHERE ICULOS IN (SELECT ICULOS FROM {} ORDER BY ICULOS
                LIMIT (
                    SELECT GREATEST(COUNT(*) - 6, 0)
                    FROM {}
                ));
        """).format(
            sql.Identifier(Patient_ID),
            sql.Identifier(Patient_ID),
            sql.Identifier(Patient_ID)
        )
    )

    conn.commit()

def data_after_FE(Patient_ID):

    df = load_data(Patient_ID)

    Missing_Handling(df)

    Add_Features(df)

    columns = df.columns

    set_clause = ", ".join(f"{col}=%s" for col in columns)

    update_query = sql.SQL("""
        UPDATE {}
        SET {}
        WHERE ICULOS=%s
    """).format(
        sql.Identifier(f"patient_cache_{Patient_ID}"),
        sql.SQL(set_clause)
    )

    values = [df.iloc[-1][c] for c in columns]
    values.append(df.iloc[-1]["ICULOS"])

    cursor.execute(update_query, values)

    conn.commit()

    return df.tail(1)


