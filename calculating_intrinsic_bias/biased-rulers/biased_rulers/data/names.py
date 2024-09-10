import re
import pandas as pd


def load_data():
    "Load gendered names from Lauscher et al. (2021) used by DisCo."
    #df = pd.read_csv("../data/name_pairs.txt", sep="\t", header=None)
    df = pd.read_csv("../data/white_hispanic_names.txt", sep="\t", header=None)
    
    return df