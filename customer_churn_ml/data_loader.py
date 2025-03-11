import pandas as pd
from db import execute_query


def load_data():
    """
    Load all the data from the customers table in a dataframe:
    """

    query = "SELECT * FROM customer;"
    result = execute_query(query=query)

    if result:
        df = pd.DataFrame(result)
        return df
    else:
        return None
    

