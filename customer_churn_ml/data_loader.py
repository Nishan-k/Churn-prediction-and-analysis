import pandas as pd
from customer_churn_ml.db import execute_query


def load_data():
    """
    Load all the data from the customers table in a dataframe:
    """

    query = "SELECT * FROM customer;"
    result, column_names = execute_query(query=query)
    
   

    if result and column_names:
        df = pd.DataFrame(result, columns=column_names)
        return df
    else:
        return None
    

def get_churn_distribution():
    """
    Get the summary of customers:
    """
    query = "SELECT churn, COUNT(*) FROM customer GROUP BY churn;"
    result, column_names = execute_query(query=query)

    if result and column_names:
        df = pd.DataFrame(result, columns=column_names)
        return df
    else:
        return None
    


def get_churn_count():
    """
    Get the total count of customer who will churn.
    """
    query = "SELECT COUNT(*) FROM customer WHERE churn = 'Yes';"
    result, column_names = execute_query(query=query)

    if result and column_names:
        df = pd.DataFrame(result, columns=column_names)
        return df.iloc[0, 0]
    else:
        return None


def get_total_customer_counts():
    """
    Returns the total number of customers.
    """
    query = "SELECT COUNT(*) FROM customer;"
    result, column_names = execute_query(query=query)

    if result and column_names:
        df = pd.DataFrame(result, columns=column_names)
        return df.iloc[0, 0]
    else:
        return None