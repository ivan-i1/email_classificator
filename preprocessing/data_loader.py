import pandas as pd
from Config import *


def get_input_data() -> pd.DataFrame:
    df = pd.read_csv("./data/AppGallery.csv", skipinitialspace=True)
    df.rename(columns={'Type 1': 'y1', 'Type 2': 'y2',
              'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].astype('U')
    df["y"] = df[Config.CLASS_COL]
    return df[df["y"].notna() & (df["y"] != "")]
