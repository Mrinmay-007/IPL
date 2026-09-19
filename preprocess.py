
import pandas as pd

def process(delv, mat):
    # Ensure the 'season' is processed correctly; extract and convert to integer
    mat['season'] = mat['season'].apply(lambda x: int(str(x)[:4]))  # Convert to integer after extracting year

    # Merge the match and delivery dataframes based on match_id and id
    df = delv.merge(mat, left_on='match_id', right_on='id', how='left')

    return df

