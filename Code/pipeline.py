import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Retrieves cleaned data from RELISH and TREC npy files
def process_data_from_tsv(file_path_in=None):
    '''
    Retrieves cleaned data from RELISH and TREC npy files, separating each column into their own respective list.

    Parameters
    ----------
    filepathIn: str
            The filepath of the RELISH or TREC input npy file.

    Returns
    -------
    pmids: list of str
            A list of all pubmed ids in the corpus.
    titles: list of str
            A list of lists where each sub-list contains the words in the cleaned/processed title.
    abstracts: list of str
            A list of lists where each sub-list contains the words in the cleaned/processed abstract.
    docs: list of str
            A list of lists where each sub-list contains the words in the cleaned/processed document (title + abstract).
    '''
    doc = np.load(file_path_in, allow_pickle=True)

    pmids = []
    titles = []
    abstracts = []
    docs = []

    for line in doc:
        pmids.append(np.ndarray.tolist(line[0]))
        titles.append(np.ndarray.tolist(line[1]))
        abstracts.append(np.ndarray.tolist(line[2]))
        docs.append(np.ndarray.tolist(line[1]) + np.ndarray.tolist(line[2]))

    return (pmids, titles, abstracts, docs)

# Create Doc2Vec Model    
def createDoc2VecModel(pmids, docs, output_file):
    '''
    Create the Doc2Vec model using Gensim for the documents in the corpus.

    Parameters
    ----------
    pmids: list of str
            A list of all pubmed ids in the corpus.
    docs: list of str
            A list of lists where each sub-list contains the words in the cleaned/processed document (title + abstract).
    output_file: str
            File path of the Doc2Vec model generated.
    '''
    tagged_data = [TaggedDocument(words=_d, tags=[str(pmids[i])]) for i, _d in enumerate(docs)]

    model = Doc2Vec(vector_size=200, window=5, min_count=1, epochs=5)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    model.save(output_file)

# Generate Document Embeddings
def create_document_embeddings(pmids, doc2vec_model, output_directory):
    '''
    Create the document embeddings for the documents in the corpus using the Doc2Vec model.

    Parameters
    ----------
    pmids: list of str
            A list of all pubmed ids in the corpus.
    doc2vec_model: str
            File path of the Doc2Vec model.
    output_directory: str
            The directory path where the document embeddings will be stored.
    '''
    model = Doc2Vec.load(doc2vec_model)

    for pmid in pmids:
        np.save(f'{output_directory}/{pmid}', model.docvecs[str(pmid)])

# Adds the 4th column to the relevance matrix file, containing the cosine similarity of the respective reference and assessed pmids.  
def update_relevance_matrix(input_file, doc2vec_model, output_file, dataset): 
    '''
    Creates the TSV file by updating the relevance matrix file and adds the 4th column, 
    consisting of the cosine similarity between the respective pmids.

    Parameters
    ----------
    input_file: str
            File path to the Relevance Matrix file.
    doc2vec_model: str
            File path of the Doc2Vec model.
    output_file: str
            Path where the output TSV file will be stored.
    dataset: str
            TREC or RELISH representing the dataset taken into consideration.
    '''
    if dataset == "RELISH":
        # Read the RELISH Relevance Matrix csv file       
        matrix_df = pd.read_csv(input_file)

        # Adds the empty 4th column to the file
        matrix_df["Cosine Similarity"] = ""

        # Load the Doc2Vec model for RELISH
        model = Doc2Vec.load(doc2vec_model)

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
        # Read the TREC Relevance Matrix tsv file
        matrix_df = pd.read_csv(input_file, delimiter='\t')

        # Adds the empty 4th column to the file
        matrix_df["Cosine Similarity"] = ""

        # Load the Doc2Vec model for TREC
        model = Doc2Vec.load(doc2vec_model)

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


