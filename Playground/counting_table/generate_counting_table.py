import math

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt 

def load_relevance_matrix(input_path: str):
    """
    Reads a .CSV file containing the relevance matrix for RELISH. Three columns
    are needed: PMID 1, PMID 2 and Relevance Assessment.
    If the matrix is not populated with Cosine Similarity, use function
    "create_random_table()" to emulate some results.
    Parameters
    ----------
    input_path : str
        File path to the Relevance Matrix. It is used to populate the forth
        column. 
    Returns
    -------
    data: pd.DataFrame
        Dataframe with 4 columns: PMID 1, PMID 2, Relevance Assessment
        and Cosine Similarity.
    """
    data = pd.read_csv(input_path, delimiter="\t")

    return data

def random_cosine_similarities(data: pd.DataFrame):
    """
    Generates a random relevance matrix. It populates the forth column (the
    Cosine Similarity) with normally distributed random numbers. 
    
    The numbers are capped between [0, 1], and both the center and the standard
    deviation can be modified form the code.
    Parameters
    ----------
    data: pd.DataFrame
        Dataframe with at least 3 columns: PMID 1, PMID 2 and Relevance
        Assessment. 
    Returns
    -------
    data: pd.DataFrame
        Dataframe with 4 columns: PMID 1, PMID 2, Relevance Assessment and
        Cosine Similarity.
    """
    splits = data["Relevance Assessment"].value_counts().to_dict()

    # Random values generated following a normal distribution capped between
    # [0, 1]
    a = np.clip(np.random.normal(0.4, 0.15, splits[0]), 0, 1)
    b = np.clip(np.random.normal(0.6, 0.1, splits[1]), 0, 1)
    c = np.clip(np.random.normal(0.75, 0.05, splits[2]), 0, 1)

    iter_a = np.nditer(a)
    iter_b = np.nditer(b)
    iter_c = np.nditer(c)
    
    for i, row in data.iterrows():
        if row["Relevance Assessment"] == 0:
            data.at[i, "Cosine Similarity"] = math.floor(next(iter_a)*100)/100
        elif row["Relevance Assessment"] == 1:
            data.at[i, "Cosine Similarity"] = math.floor(next(iter_b)*100)/100
        elif row["Relevance Assessment"] == 2:
            data.at[i, "Cosine Similarity"] = math.floor(next(iter_c)*100)/100

    
    return data

def count_entries(data: pd.DataFrame, interval: float):
    """
    Counts the number of Relevance Assessment for a given value of Cosine
    Similarity.
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe with 4 columns: PMID 1, PMID 2, Relevance Assessment
        and Cosine Similarity.
    interval : float
        Value of Cosine Similarity to count the entries.
    Returns
    -------
    counter: dict
        Dictionary containing the counts for each Relevance Assessment
    count_explanation: str
        Counting Explanation by specifying the row numbers from the input 
        dataframe
    """
    filtered_df = data[data["Cosine Similarity"] == interval]["Relevance Assessment"]
    counter = {0: sum(filtered_df == 0), 1: sum(filtered_df == 1), 2: sum(filtered_df == 2)}

    count_explanation = "Rows: " + ", ".join(str(e) for e in list(filtered_df.index))

    return counter, count_explanation

def create_counting_table(data: pd.DataFrame):
    """
    Creates the "counting table" from a given Relevance matrix.
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe with 4 columns: PMID 1, PMID 2, Relevance Assessment
        and Cosine Similarity.
    Returns
    -------
    counting_df: pd.DataFrame
        DataFrame of the counting table generated with five columns: Cosine
        Interval, 2s, 1s, 0s and Counting Explanation.
    """
    counting_df = pd.DataFrame({"Cosine Interval":  np.round(np.linspace(0, 1, 101), 2).tolist(), "2s": 0, "1s": 0, "0s": 0, 
                                "Counting Explanation": ""})

    for i, row in counting_df.iterrows():
        interval = row["Cosine Interval"]
        interval_counts, count_explanation = count_entries(data, interval)

        counting_df.at[i, "2s"] = interval_counts[2]
        counting_df.at[i, "1s"] = interval_counts[1]
        counting_df.at[i, "0s"] = interval_counts[0]
        
        counting_df.at[i, "Counting Explanation"] = count_explanation
        
    return counting_df

def save_table(counting_df: pd.DataFrame, output_path: str):
    """
    Saves the counting table into .TSV format.
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame of the counting table with four columns: Cosine
        Interval, 2s, 1s and 0s.
    output_path : str
        Output path to where the counting table will be saved.
    """
    counting_df.to_csv(output_path, index=False, sep="\t")