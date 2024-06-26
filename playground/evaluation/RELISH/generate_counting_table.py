import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 

def load_relevance_matrix(input_path: str) -> pd.DataFrame:
    """
    Reads a .CSV file containing the relevance matrix for RELISH. Three columns
    are needed: PMID 1, PMID 2 and Relevance Assessment.
    If the matrix is not populated with Cosine Similarity, use function
    "create_random_table()" to emulate some results.
    Parameters
    ----------
    input_path : str
        File path to the Relevance Matrix. It will be used to populate the fourth
        column. 
    Returns
    -------
    data: pd.DataFrame
        Dataframe with 3 columns: PMID 1, PMID 2, Relevance Assessment.
    """
    data = pd.read_csv(input_path)

    return data

def random_cosine_similarities(data: pd.DataFrame) -> pd.DataFrame:
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

def count_entries(data: pd.DataFrame, interval: float) -> dict:
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
    """
    filtered_df = data[data["Cosine Similarity"] == interval]["Relevance Assessment"]
    counter = {0: sum(filtered_df == 0), 1: sum(filtered_df == 1), 2: sum(filtered_df == 2)}

    return counter

def create_counting_table(data: pd.DataFrame) -> pd.DataFrame:
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
        DataFrame of the counting table generated with four columns: Cosine
        Interval, 2s, 1s, 0s.
    """
    counting_df = pd.DataFrame({"Cosine Interval":  np.round(np.linspace(0, 1, 101), 2).tolist(), "2s": 0, "1s": 0, "0s": 0})

    for i, row in counting_df.iterrows():
        interval = row["Cosine Interval"]
        interval_counts = count_entries(data, interval)

        counting_df.at[i, "2s"] = interval_counts[2]
        counting_df.at[i, "1s"] = interval_counts[1]
        counting_df.at[i, "0s"] = interval_counts[0]
        
    return counting_df

def save_table(counting_df: pd.DataFrame, output_path: str) -> None:
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

def plot_graph(input_path: str, output_path: str, normalize: bool = False) -> None:
    """
    Plots the graph of "Relevance counting" against the "Cosine intervals"
    for the number of 2s, 1s and 0s in the counting table.
    Parameters
    ----------
    input_path : str
        Input path to the .TSV file containing the counting table.
    output_path : str
        Output path to where the plot will be saved.
    normalize : bool
        Boolean to determine whether the data should be normalized or not.
    """
    data = pd.read_csv(input_path, delimiter="\t", usecols=["Cosine Interval", "2s", "1s", "0s"])
    intervals = data["Cosine Interval"].values.tolist()
    two_points = data["2s"].values.tolist()
    one_points = data["1s"].values.tolist()
    zero_points = data["0s"].values.tolist()

    if normalize:
        plt.plot(intervals, [i/sum(two_points) for i in two_points], 'r', label='2 counts')  
        plt.plot(intervals, [i/sum(one_points) for i in one_points], 'b', label='1 counts') 
        plt.plot(intervals, [i/sum(zero_points) for i in zero_points], 'g', label='0 counts')
    else:
        plt.plot(intervals, two_points, 'r', label='2 counts')  
        plt.plot(intervals, one_points, 'b', label='1 counts') 
        plt.plot(intervals, zero_points, 'g', label='0 counts')

    plt.xlabel("Cosine intervals")
    plt.ylabel("Relevance counting")

    plt.legend()
    plt.savefig(output_path)
    plt.show()

