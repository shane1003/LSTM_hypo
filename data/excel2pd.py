"""data related fuctions
"""

import pandas as pd

def data_load(ID):
    """
    Load the data from excel using pandas package.

    Args : 
        id : patient id(excel file sheet name)

    Returns :
        data from id named sheet(pandas)
    """
    data = pd.read_excel("./data/data_temp.xlsx", sheet_name = str(ID)).drop(labels=['ID','Time',], axis = 1)
    
    return data



