import pandas as pd
import numpy as np
import glob

def get_pmids(file_path):
    # Load pmids from .npy file
    doc = np.load(file_path, allow_pickle=True)
    
    # Get pmids
    pmids = []
    for line in doc:
        pmids.append(np.ndarray.tolist(line[0]))

    # Convert all values in the list "pmids" from type 'str' to 'int'
    pmids = list(map(int, pmids))
    # Sort pmids in ascending order
    pmids.sort()

    return pmids

def get_embeddings(embeddings_path, pmids):
    embeddings_list = []
    for pmid in pmids:
        # Load the embeddings
        file_name = embeddings_path + str(pmid) + ".npy"
        embeddings = np.load(file_name)
        # Append the embeddings to the "embeddings_list"
        embeddings_list.append(embeddings)

    return embeddings_list

def save_dataframe(pmids, embeddings_list, output_file):
    # Create pandas dataframe with pmids and embeddings
    dict = {'pmids': pmids, 'embeddings': embeddings_list} 
    df = pd.DataFrame(dict)

    # Save dataframe to Python pickle format
    df.to_pickle(output_file)

def load_dataframe(file_path):
    # Load dataframe
    df = pd.read_pickle(file_path)
    print (df)

pmids = get_pmids('Data/TREC/TSV/TREC_documents_pruned.npy')
embeddings_list = get_embeddings('Embeddings/TREC/', pmids)
save_dataframe(pmids, embeddings_list, 'Embeddings/trec_embeddings.pkl')
load_dataframe('Embeddings/trec_embeddings.pkl')