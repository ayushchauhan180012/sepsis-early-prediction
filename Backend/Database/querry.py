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
        return "Database already exists and shifted access to the database"
    

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
    conn.commit()

def Patient_Cache_Manage(Patient_ID: str, values: list):
    cursor.execute(
        sql.SQL("""
            INSERT INTO {} (HR, O2Sat, SBP, MAP, Resp, Temp, Lactate, WBC, Creatinine, Platelets, ICULOS)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);""").format(sql.Identifier(Patient_ID)),values)
    
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

def Update_Patient_Cache(Patient_ID: str, values: list, set_clause: tuple | None = None):
    cursor.execute(
        sql.SQL("""
            UPDATE {}
            SET {}
            WHERE ICULOS = %s
        """).format(
            sql.Identifier(f"patient_cache_{Patient_ID}"),
            sql.SQL(set_clause)
        ),
        values
    )

    conn.commit()