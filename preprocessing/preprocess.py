import pandas as pd
import re
from Config import *
from preprocessing.noise_remover import *
from preprocessing.cleaner import *
from preprocessing.translation import *

def preprocess_data(df):
    # De-duplicate input data
    df = de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df
