import pandas as pd
from gensim.models.doc2vec import Doc2Vec

# Adds the 4th column to the relevance matrix file, containing the cosine similarity of the respective reference and assessed pmids.  
def update_relevance_matrix(input_file: str, model: Doc2Vec, output_file: str, dataset: str) -> None: 
    """
    Updates the relevance matrix tsv file and adds the 4th column, 
    consisting of the cosine similarity between the respective pmids.

    Parameters
    ----------
    input_file: str
            File path to the Relevance Matrix tsv file.
    model: Doc2Vec
            Doc2Vec model.
    output_file: str
            Path where the output tsv file will be stored.
    dataset: str
            TREC or RELISH representing the dataset taken into consideration.
    """
    # Read the Relevance Matrix tsv file
    matrix_df = pd.read_csv(input_file, delimiter='\t')

    # Adds the empty 4th column to the file
    matrix_df["Cosine Similarity"] = ""

    if dataset == "RELISH":
        for index, row in matrix_df.iterrows():
            ref_pmid = row["PMID Reference"]
            assessed_pmid = row["PMID Assessed"]

            try:
                # Determine the cosine similarity of the ref and assessed pmids and add to the 4th column
                row["Cosine Similarity"] = round(model.docvecs.similarity(str(ref_pmid), str(assessed_pmid)), 2)
            except:
                # Leave the 4th column empty if the ref or assessed pmid not found in the dataset
                row["Cosine Similarity"] = ""

            # Make changes in the original dataframe
            matrix_df.at[index,'Cosine Similarity'] = row['Cosine Similarity']

    elif dataset == "TREC":
        for index, row in matrix_df.iterrows():
            ref_pmid = row["PMID1"]
            assessed_pmid = row["PMID2"]

            try:
                # Determine the cosine similarity of the ref and assessed pmids and add to the 4th column
                row["Cosine Similarity"] = round(model.docvecs.similarity(str(ref_pmid), str(assessed_pmid)), 2)
            except:
                # Leave the 4th column empty if the ref or assessed pmid not found in the dataset
                row["Cosine Similarity"] = ""

            # Make changes in the original dataframe
            matrix_df.at[index,'Cosine Similarity'] = row['Cosine Similarity']
                
    matrix_df.to_csv(output_file, index=False, sep="\t")