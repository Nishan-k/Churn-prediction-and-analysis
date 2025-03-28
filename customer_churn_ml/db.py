import psycopg2
from psycopg2 import sql
from customer_churn_ml.credentials import db_config



def execute_query(query):
    """"
    Creates DB connection, executes the query, and returns the result.
    """

    try:
        connection = psycopg2.connect(**db_config)
        
        cursor = connection.cursor()

        cursor.execute(query)
        result = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        return result, column_names
    
    except psycopg2.Error as e:
        print(f"An error has occured: {e}")
        return None
    
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()



