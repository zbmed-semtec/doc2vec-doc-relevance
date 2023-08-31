import pandas as pd

def load_trec_relevance_pairs(input_path: str) -> pd.DataFrame:
    trec_df = pd.read_csv(input_path, delimiter="\t")

    return trec_df

def get_trec_repurposed_with_topic(trec_df: pd.DataFrame) -> pd.DataFrame:
    trec_df['Rel-d2d'] = trec_df[['Rel1','Rel2']].min(axis=1)
    trec_df = trec_df.drop('Rel1', axis=1)
    trec_df = trec_df.drop('Rel2', axis=1)

    return trec_df

def get_trec_repurposed_data(trec_df: pd.DataFrame) -> pd.DataFrame:
    trec_df = trec_df.sort_values(['PMID1', 'PMID2', 'Rel-d2d']).drop_duplicates(['PMID1', 'PMID2'], keep='last')
    trec_df = trec_df.drop('Topic', axis=1)

    return trec_df

def save_file(trec_df: pd.DataFrame, output_path: str) -> None:
    trec_df.to_csv(output_path, index=False, sep="\t")


input_path = "Data/TREC/Relevance_Matrix/trec_relevance_pairs.tsv"
trec_df = load_trec_relevance_pairs(input_path)

trec_df = get_trec_repurposed_with_topic(trec_df)
trec_df = get_trec_repurposed_data(trec_df)

output_path = "Data/TREC/Relevance_Matrix/trec_repurposed_matrix.tsv"
save_file(trec_df, output_path)