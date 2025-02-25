#Methods related to data loading and all pre-processing steps will go here
import pandas as pd
def get_input_data():
    try:
        return pd.read_csv('input.csv')
    except():
        return pd.read_csv('AppGallery.csv')
