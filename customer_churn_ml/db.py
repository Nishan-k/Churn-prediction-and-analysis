import psycopg2
from psycopg2 import sql
from credentials import db_config



def execute_query(query):
    """"
    Creates DB connection, executes the query, and returns the result.
    """

    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()

        cursor.execute(query)
        result = cursor.fetchall()

        return result
    
    except psycopg2.Error as e:
        print(f"An error has occured: {e}")
        return None
    
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()