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
    

def churn_count():
    """
    Get the current count of the Churn:
    """
    query = "SELECT churn, COUNT(*) FROM customer GROUP BY churn;"
    result, column_names = execute_query(query=query)

    if result and column_names:
        df = pd.DataFrame(result, columns=column_names)
        return df
    else:
        return None