import itertools
import pandas as pd

def load_TREC_data(input_path: str) -> pd.DataFrame:
    trec_df = pd.read_csv(input_path, delimiter="\t", names = ["Topic", "Zeros", "PMID", "Relevance"], usecols=["Topic", "PMID", "Relevance"])

    # Replace all relevance with value of 2 with the value of 1
    trec_df.loc[trec_df["Relevance"] == 2, "Relevance"] = 1

    return trec_df

def get_unique_topics(trec_df: pd.DataFrame) -> list:
    # Get unique topics
    topics = trec_df["Topic"].unique().tolist()

    return topics

def determine_group(relevance_pairs: list) -> list:
    # [1,1] = "A"; [1,0] or [0,1] = "B"; [0,0] = "C"
    group_dict = {2: "A", 1: "B", 0: "C"} # keys based on the sum of pairs
    for index, pair in enumerate(relevance_pairs):
        pair = list(pair)
        pair_sum = sum(pair)
        group = group_dict[pair_sum]

        pair.append(group)
        pair = tuple(pair)
        relevance_pairs[index] = pair

    return relevance_pairs
    
def prepare_data(trec_df: pd.DataFrame, topics: list) -> list:
    output_data = []

    # Iterate over topics
    for topic in topics:
        # Get rows for this particular topic
        topic_df = trec_df.loc[trec_df['Topic'] == topic]

        # Get list of pmid pairs
        pmid_pairs = topic_df["PMID"].values.tolist()
        pmid_pairs = list(itertools.combinations(pmid_pairs, 2))

        # Get list of relevance assessment pairs
        relevance_pairs = topic_df["Relevance"].values.tolist()
        relevance_pairs = list(itertools.combinations(relevance_pairs, 2))

        # Add group to relevance assessment pairs in the relevance_pairs list
        relevance_pairs = determine_group(relevance_pairs)

        # Combine the above two list element-wise
        combined_pairs = [i + j for i, j in zip(pmid_pairs, relevance_pairs)]

        # Convert sub-element type from tuple to list
        combined_pairs = [list(s) for s in combined_pairs]

        # Append the topic to each sub-list inside combined_pairs
        combined_pairs = [[topic] + sub for sub in combined_pairs]

        output_data += combined_pairs

    return output_data

def save_file(output_data: list, output_path: str) -> None:
    output_df = pd.DataFrame(output_data, columns=["Topic", "PMID1", "PMID2", "SRel1", "SRel2", "Group"])
    output_df.to_csv(output_path, index=False, sep="\t")


input_path = "Data/TREC/Relevance_Matrix/TREC.tsv"
trec_df= load_TREC_data(input_path)

topics = get_unique_topics(trec_df)
output_data = prepare_data(trec_df, topics)

output_path = "Data/TREC/Relevance_Matrix/trec_simplified_relevance_matrix.tsv"
save_file(output_data, output_path)
