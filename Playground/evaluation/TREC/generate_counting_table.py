import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 

def load_relevance_matrix(input_path: str) -> pd.DataFrame:
    data = pd.read_csv(input_path, delimiter='\t')

    return data

def random_cosine_similarities(data: pd.DataFrame) -> pd.DataFrame:
    splits = data["Group"].value_counts().to_dict()

    a = np.clip(np.random.normal(0.4, 0.15, splits['A']), 0, 1)
    b = np.clip(np.random.normal(0.6, 0.1, splits['B']), 0, 1)
    c = np.clip(np.random.normal(0.75, 0.05, splits['C']), 0, 1)

    iter_a = np.nditer(a)
    iter_b = np.nditer(b)
    iter_c = np.nditer(c)

    for i, row in data.iterrows():
        if row["Group"] == 'A':
            data.at[i, "Cosine Similarity"] = math.floor(next(iter_a)*100)/100
        elif row["Group"] == 'B':
            data.at[i, "Cosine Similarity"] = math.floor(next(iter_b)*100)/100
        elif row["Group"] == 'C':
            data.at[i, "Cosine Similarity"] = math.floor(next(iter_c)*100)/100

    return data

def get_unique_topics(data: pd.DataFrame) -> list:
    topics = data['Topic'].unique().tolist()

    return topics

def count_entries(data: pd.DataFrame, interval: float) -> dict:
    filtered_df = data[data["Cosine Similarity"] == interval]["Group"]
    counter = {'A': sum(filtered_df == 'A'), 'B': sum(filtered_df == 'B'), 'C': sum(filtered_df == 'C')}

    return counter

def create_counting_table(data: pd.DataFrame) -> pd.DataFrame:
    counting_df = pd.DataFrame({"Cosine Interval":  np.round(np.linspace(0, 1, 101), 2).tolist(), "As": 0, "Bs": 0, "Cs": 0})

    for i, row in counting_df.iterrows():
        interval = row["Cosine Interval"]
        interval_counts = count_entries(data, interval)

        counting_df.at[i, "As"] = interval_counts['A']
        counting_df.at[i, "Bs"] = interval_counts['B']
        counting_df.at[i, "Cs"] = interval_counts['C']
        
    return counting_df

def save_table(counting_df: pd.DataFrame, output_path: str) -> None:
    counting_df.to_csv(output_path, index=False, sep="\t")

def plot_graph(input_path: str, output_path: str, normalize: bool = False) -> None:
    data = pd.read_csv(input_path, delimiter="\t", usecols=["Cosine Interval", "As", "Bs", "Cs"])
    intervals = data["Cosine Interval"].values.tolist()

    a_points = data["As"].values.tolist()
    b_points = data["Bs"].values.tolist()
    c_points = data["Cs"].values.tolist()

    plt.figure()

    if normalize:
        plt.plot(intervals, [i/sum(a_points) for i in a_points], 'r', label='A counts')  
        plt.plot(intervals, [i/sum(b_points) for i in b_points], 'b', label='B counts') 
        plt.plot(intervals, [i/sum(c_points) for i in c_points], 'g', label='C counts')
    else:
        plt.plot(intervals, a_points, 'r', label='A counts')  
        plt.plot(intervals, b_points, 'b', label='B counts') 
        plt.plot(intervals, c_points, 'g', label='C counts')

    plt.xlabel("Cosine intervals")
    plt.ylabel("Relevance counting")

    plt.legend()
    plt.savefig(output_path)
    plt.show()

def create_counting_table_by_topic(data: pd.DataFrame) -> None:
    topics = get_unique_topics(data)

    for topic in topics:
        output_path = "Playground/evaluation/TREC/tsv_files/"
        plot_path = "Playground/evaluation/TREC/plots/"

        topic_df = data[data['Topic'] == topic]
        counting_df = create_counting_table(topic_df)

        output_path += "counting_table_by_topic_" + str(topic) + ".tsv"
        save_table(counting_df, output_path)

        plot_path +=  "output_plot_by_topic_" + str(topic) + ".png"
        plot_graph(output_path, plot_path)

        statement = "Counting completed for topic " + str(topic)
        print (statement)

