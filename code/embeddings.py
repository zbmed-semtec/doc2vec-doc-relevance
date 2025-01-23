import os
import yaml
import argparse
import itertools
import numpy as np
import pandas as pd
from typing import Union, List
import embeddings_dataframe as ed
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Retrieves cleaned data from RELISH and TREC npy files
def process_data_from_npy(file_path_in: str = None) -> Union[List[str], List[List[str]], List[List[str]], List[List[str]]]:
    """
    Retrieves cleaned data from RELISH and TREC npy files, separating each column 
    into their own respective list.

    Parameters
    ----------
    filepathIn: str
            The filepath of the RELISH or TREC input npy file.
    Returns
    -------
    pmids: List[str]
            A list of all pubmed ids in the corpus.
    titles: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed title.
    abstracts: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed abstract.
    docs: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed document (title + abstract).
    """
    doc = np.load(file_path_in, allow_pickle=True)

    pmids = []
    article_docs = []

    for line in range(len(doc)):
        pmids.append(int(doc[line][0]))
        if isinstance(doc[line][1], (np.ndarray, np.generic)):
            article_docs.append(np.ndarray.tolist(doc[line][1]))
            article_docs[line].extend(np.ndarray.tolist(doc[line][2]))
        else:
            article_docs.append(doc[line][1])
            article_docs[line].extend(doc[line][2])
    return (pmids, article_docs)

def generate_param_combinations(params):
    param_keys = []
    param_values = []
    
    for key, value in params.items():
        if 'values' in value:  # Check if 'values' exist in this parameter
            param_keys.append(key)
            param_values.append(value['values'])
        else:
            param_keys.append(key)
            param_values.append([value['value']])  # Use the single value as a list
    
    param_combinations = [dict(zip(param_keys, combination)) 
                          for combination in itertools.product(*param_values)]
    
    return param_combinations


# Create and train the Doc2Vec Model
def createDoc2VecModel(pmids: List[str], docs: List[List[str]], params: dict) -> Doc2Vec:
    """
    Create and train the Doc2Vec model using Gensim for the documents 
    in the corpus.

    Parameters
    ----------
    pmids: List[str]
            A list of all pubmed ids in the corpus.
    docs: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed document (title + abstract).
    params: dict
            Dictionary containing the parameters for the Doc2Vec model.
    Returns
    -------
    model: Doc2Vec
            Doc2Vec model.
    """
    tagged_data = [TaggedDocument(words=_d, tags=[str(pmids[i])])
                   for i, _d in enumerate(docs)]

    # model = Doc2Vec(vector_size=200, window=5, min_count=1, epochs=5)
    model = Doc2Vec(**params)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count,
                epochs=model.epochs)

    return model

# Save the Doc2Vec Model
def saveDoc2VecModel(model: Doc2Vec, output_file: str) -> None:
    """
    Saves the Doc2Vec model.

    Parameters
    ----------
    model: Doc2Vec
            Doc2Vec model.
    output_file: str
            File path of the Doc2Vec model generated.
    """
    model.save(output_file)

# Generate and save the document embeddings
def generate_document_embeddings(pmids: List[str], model: Doc2Vec, iteration: int, output_path: str) -> None:
        """
        Create and save the document embeddings for the documents 
        in the corpus using the Doc2Vec model.

        Parameters
        ----------
        pmids: list of str
                A list of all pubmed ids in the corpus.
        model: Doc2Vec
                Doc2Vec model.
        iteration: int
                Hyperparameter configuration number.
        output_directory: str
                The directory path where the document embeddings 
                will be stored.
        """
        document_embeddings = []
        for pmid in pmids:
                vector = model.docvecs[str(pmid)]
                document_embeddings.append(vector)    
        data = {"pmids": pmids, "embeddings": document_embeddings}
        embeddings_df = pd.DataFrame(data)
        embeddings_df = embeddings_df.sort_values("pmids")
        
        os.makedirs(f"{output_path}", exist_ok=True)
        embeddings_df.to_pickle(f'{output_path}/embeddings_{iteration}.pkl') 
        
if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--input", type=str,
                                help="Path to input RELISH tokenized .npy file")
        parser.add_argument("-o", "--output", type=str,
                                help="Path to save embeddings pickle file")
        parser.add_argument("-p", "--params", type=str,
                                help="Path to hyperparameter yaml file.")
        args = parser.parse_args()

        params = []
        with open(args.params, "r") as file:
                content = yaml.safe_load(file)
                params = content['params']

        param_combinations = generate_param_combinations(params)
        model_output_file_base = "./data/models/doc2vec_model"
        model_output_dir = os.path.dirname(model_output_file_base)
        if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)

        pmids, docs = process_data_from_npy(args.input)
        for i, param_set in enumerate(param_combinations):
                print(f"Training model with hyperparameters: {param_set}")
                model = createDoc2VecModel(pmids, docs, param_set)
                model_output_file = f"{model_output_file_base}_{i}"
                saveDoc2VecModel(model, model_output_file)
                generate_document_embeddings(pmids, model, i, args.output)