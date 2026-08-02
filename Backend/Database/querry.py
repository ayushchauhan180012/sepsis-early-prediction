import psycopg
from psycopg2 import sql

conn = psycopg.connect(
    host="",
    port=5432,
    dbname="postgres",
    user="",
    password="2407"
)

cursor = conn.cursor

def check_db_exist(db_name):
    cursor.execute(sql.SQL(""" SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = {});""").format(sql.Identifier(db_name)),)

    return cursor.fetchone()[0]

def Create_db(db_name):
    if not check_db_exist(db_name):
        cursor.execute(sql.SQL("""CREATE DATABASE {};""").format(sql.Identifier(db_name)))
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
        return "Database Created Successfully"
    else:
        return "Database already exists"

def check_Patient_exists(Patient_ID):
    cursor.execute("""SELECT EXISTS ( SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s);""",(Patient_ID,))

    return cursor.fetchone()[0]

def Create_Patient(Patient_ID: str ,Details: dict):
    cursor.execute(sql.SQL("""
    CREATE TABLE {} (
        HR FLOAT,
        O2Sat FLOAT,
        SBP FLOAT,
        MAP FLOAT,
        Resp FLOAT,
        Temp FLOAT,
        Lactate FLOAT,
        WBC FLOAT,
        Creatinine FLOAT,
        Platelets FLOAT,
        Age INT DEFAULT %s,
        ICULOS INT NOT NULL
    );
    """).format(sql.Identifier(Patient_ID)),(Details['age'],)
    )