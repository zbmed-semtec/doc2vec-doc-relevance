import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 

def load_relevance_matrix(input_path: str) -> pd.DataFrame:
    data = pd.read_csv(input_path, delimiter='\t')

    return data

def random_cosine_similarities(data: pd.DataFrame) -> pd.DataFrame:
    splits = data["Group"].value_counts().to_dict()

    a = np.clip(np.random.normal(0.8, 0.2, splits['A']), 0, 1)
    b = np.clip(np.random.normal(0.75, 0.15, splits['B']), 0, 1)
    c = np.clip(np.random.normal(0.6, 0.15, splits['C']), 0, 1)
    d = np.clip(np.random.normal(0.45, 0.1, splits['D']), 0, 1)
    e = np.clip(np.random.normal(0.4, 0.05, splits['E']), 0, 1)
    f = np.clip(np.random.normal(0.3, 0.05, splits['F']), 0, 1)

    iter_a = np.nditer(a)
    iter_b = np.nditer(b)
    iter_c = np.nditer(c)
    iter_d = np.nditer(d)
    iter_e = np.nditer(e)
    iter_f = np.nditer(f)

    for i, row in data.iterrows():
        if row["Group"] == 'A':
            data.at[i, "Cosine Similarity"] = math.floor(next(iter_a)*100)/100
        elif row["Group"] == 'B':
            data.at[i, "Cosine Similarity"] = math.floor(next(iter_b)*100)/100
        elif row["Group"] == 'C':
            data.at[i, "Cosine Similarity"] = math.floor(next(iter_c)*100)/100
        elif row["Group"] == 'D':
            data.at[i, "Cosine Similarity"] = math.floor(next(iter_d)*100)/100
        elif row["Group"] == 'E':
            data.at[i, "Cosine Similarity"] = math.floor(next(iter_e)*100)/100
        elif row["Group"] == 'F':
            data.at[i, "Cosine Similarity"] = math.floor(next(iter_f)*100)/100

    return data

def get_unique_topics(data: pd.DataFrame) -> list:
    topics = data['Topic'].unique().tolist()

    return topics

def count_entries(data: pd.DataFrame, interval: float) -> dict:
    filtered_df = data[data["Cosine Similarity"] == interval]["Group"]
    counter = {'A': sum(filtered_df == 'A'), 'B': sum(filtered_df == 'B'), 'C': sum(filtered_df == 'C'), 
               'D': sum(filtered_df == 'D'), 'E': sum(filtered_df == 'E'), 'F': sum(filtered_df == 'F')}

    return counter

def create_counting_table(data: pd.DataFrame) -> pd.DataFrame:
    counting_df = pd.DataFrame({"Cosine Interval":  np.round(np.linspace(0, 1, 101), 2).tolist(),
                                "As": 0, "Bs": 0, "Cs": 0, "Ds": 0, "Es": 0, "Fs": 0})

    for i, row in counting_df.iterrows():
        interval = row["Cosine Interval"]
        interval_counts = count_entries(data, interval)

        counting_df.at[i, "As"] = interval_counts['A']
        counting_df.at[i, "Bs"] = interval_counts['B']
        counting_df.at[i, "Cs"] = interval_counts['C']
        counting_df.at[i, "Ds"] = interval_counts['D']
        counting_df.at[i, "Es"] = interval_counts['E']
        counting_df.at[i, "Fs"] = interval_counts['F']
        
    return counting_df

def save_table(counting_df: pd.DataFrame, output_path: str) -> None:
    counting_df.to_csv(output_path, index=False, sep="\t")

def plot_graph(input_path: str, output_path: str) -> None:
    data = pd.read_csv(input_path, delimiter="\t", usecols=["Cosine Interval", "2s", "1s", "0s"])
    intervals = data["Cosine Interval"].values.tolist()

    a_points = data["As"].values.tolist()
    b_points = data["Bs"].values.tolist()
    c_points = data["Cs"].values.tolist()
    d_points = data["Ds"].values.tolist()
    e_points = data["Es"].values.tolist()
    f_points = data["Fs"].values.tolist()

    plt.plot(intervals, a_points, 'r', label='A counts')  
    plt.plot(intervals, b_points, 'b', label='B counts') 
    plt.plot(intervals, c_points, 'g', label='C counts')
    plt.plot(intervals, d_points, 'c', label='D counts')  
    plt.plot(intervals, e_points, 'm', label='E counts') 
    plt.plot(intervals, f_points, 'k', label='F counts')

    plt.xlabel("Cosine intervals")
    plt.ylabel("Relevance counting")

    plt.legend()
    plt.savefig(output_path)
    plt.show()

def create_counting_table_by_topic(data: pd.DataFrame) -> None:
    topics = get_unique_topics(data)

    output_path = "Playground/evaluation/TREC/tsv_files/"
    plot_path = "Playground/evaluation/TREC/plots/"

    for topic in topics:
        topic_df = data[data['Topic'] == topic]
        counting_df = create_counting_table(topic_df)

        output_path += "counting_table_by_topic_" + str(topic) + ".tsv"
        save_table(counting_df, output_path)

        plot_path +=  "output_plot_by_topic_" + str(topic) + ".png"
        plot_graph(output_path, plot_path)

