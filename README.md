# Doc2Vec-Doc-relevance
This repository focuses on an approach exploring and assessing literature-based doc-2-doc recommendations using the Doc2Vec technique with its application to the RELISH dataset. 


## Table of Contents

1. [About](#about)
2. [Input Data](#input-data)
3. [Pipeline](#pipeline)
    1. [Generate Embeddings](#generate-embeddings)
        - [Create Tagged Documents](#create-tagged-documents)
        - [Generate and Train Doc2Vec models](#generate-and-train-doc2vec-models)
          - [Parameters](#parameters)
    2. [Format embeddings](#format-embeddings)
    3. [Calculate Cosine Similarity](#calculate-cosine-similarity)
    4. [Hyperparameter Optimization](#hyperparameter-optimization)
    5. [Evaluation](#evaluation)
        - [Precision@N](#precisionn)
        - [nDCG@N](#ndcgn)
7. [Code Implementation](#code-implementation)
8. [Getting Started](#getting-started)
9. [Tutorial](#tutorial)



## About

 Our approach involves employing the [doc2vec](https://arxiv.org/pdf/1405.4053v2.pdf) model, which extends the popular word2vec technique to capture document-level semantics. By encoding documents and their textual content into fixed-length vectors, doc2vec facilitates similarity calculations and enables meaningful comparisons between documents. This approach is harnessed to derive insightful doc-2-doc recommendations within the realm of biomedical research, specifically employing the RELISH dataset. In order to do so, we employ the [doc2vec model](https://radimrehurek.com/gensim/models/doc2vec.html) from the [Gensim](https://radimrehurek.com/gensim/index.html) library.

## Input Data

The input data for this method consists of preprocessed tokens derived from the RELISH documents. These tokens are stored in the **RELISH.npy file**, which contains preprocessed arrays comprising PMIDs, document titles, and abstracts. These arrays are generated through an extensive preprocessing pipeline, as elaborated in the [relish-preprocessing repository](https://github.com/zbmed-semtec/relish-preprocessing). Within this preprocessing pipeline, both the title and abstract texts undergo several stages of refinement: structural words are eliminated, text is converted to lowercase, and finally, tokenization is employed, resulting in arrays of individual words.

## Pipeline

This section outlines the progression from generating document embeddings to conducting hyperparameter optimization and ultimately evaluating the effectiveness of the approach.

### Generate Embeddings
The following section outlines the process of generating document-level embeddings for each PMID of the RELISH corpus.

#### Create Tagged Documents 
In this initial step, we create  `TaggedDocuments `, which associate each PMID with a corresponding list of words. Here, we combine the abstract and title of each document into a unified paragraph (or document). This unified text serves as the input for our Doc2Vec model, allowing it to capture the semantic meaning of the entire document.
#### Generate and Train Doc2Vec models 
In the second phase, we construct and train Doc2Vec models with customizable hyperparameters. These models are designed to understand the relationships between documents and words in a high-dimensional vector space. We employ the parameters shown below in order to generate our models.
##### Parameters

+ **dm:** {1,0} Refers to the training algorithm. If dm=1, distributed memory is used otherwise, distributed bag of words is used.
+ **vector_size:** It represents the dimensions of the generated embeddings, with options of 200, 300 and 400 in our case.
+ **window:** Represents the maximum distance between the current and predicted word, with values fof 5,6 and 7 in our case.
+ **epochs:** Refers to the number of iterations over the training dataseta and is set at 15 in this context.
+ **min_count:** It is the minimum number of appearances a word must have to not be ignored by the algorithm and is configured at a minimum of 5.

### Format embeddings
After model training, we can extract document-level embeddings. These embeddings are numerical vectors that represent the content and context of each document in a continuous vector space. These embeddings are stored by the model, associated with each PMID. For further downstream document similarity calculations, we format and save these embeddings for each document with its PMID as a dataframe in a pickle file. Each specific set of hyperparameter combination results in having a separate pickle file.

## Calculate Cosine Similarity
To assess the similarity between two documents within the RELISH corpus, we employ the Cosine Similarity metric. This process enables the generation of a 4-column matrix containing cosine similarity scores for existing pairs of PMIDs within our corpus. For a more detailed explanation of the process, please refer to this [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Cosine_Similarity).

## Hyperparameter Optimization
*To be written*

## Evaluation

### Precision@N

In order to evaluate the effectiveness of this approach, we make use of Precision@N. Precision@N measures the precision of retrieved documents at various cutoff points (N).We generate a Precision@N matrix for existing pairs of documents within the RELISH corpus, based on the original RELISH JSON file. The code determines the number of true positives within the top N pairs and computes Precision@N scores. The result is a Precision@N matrix with values at different cutoff points, including average scores. For detailed insights into the algorithm, please refer to this [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Precision%40N_existing_pairs).


### nDCG@N

Another metric used is the nDCG@N (normalized Discounted Cumulative Gain). This ranking metric assesses document retrieval quality by considering both relevance and document ranking. It operates by using a TSV file containing relevance and cosine similarity scores, involving the computation of DCG@N and iDCG@N scores. The result is an nDCG@N matrix for various cutoff values (N) and each PMID in the corpus, with detailed information available in the [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Evaluation).

## Code Implementation


The [`run_embeddings.py`](./code/generate_embeddings/run_embeddings.py) serves as a comprehensive wrapper function, supporting the creation of tagged documents, model generation, training, embedding generation, and the subsequent storage of these embeddings as pickle files. Individual functions for each task are provided in the other two code scripts:

+ [`embeddings.py`](./code/generate_embeddings/embeddings.py) : Creation of tagged documents from input tokens, creation and training of Doc2Vec models, generation of embeddings. 
+ [`embeddings.dataframe.py`](./code/generate_embeddings/embeddings_dataframe.py) : Creates a dataframe of embeddings with its corresponding PMID, sorts and stores it as a pickle file.


## Getting Started

To get started with this project, follow these steps:

### Step 1: Clone the Repository
First, clone the repository to your local machine using the following command:

###### Using HTTP:

```
git clone https://github.com/zbmed-semtec/doc2vec-doc-relevance.git
```

###### Using SSH:
Ensure you have set up SSH keys in your GitHub account.

```
git clone git@github.com:zbmed-semtec/doc2vec-doc-relevance.git
```

### Step 2: Create a virtual environment and install dependencies

To create a virtual environment within your repository, run the following command:

```
python3 -m venv .venv 
source .venv/bin/activate   # On Windows, use '.venv\Scripts\activate' 
```

To confirm if the virtual environment is activated and check the location of yourPython interpreter, run the following command:

```
which python    # On Windows command prompt, use 'where python'
                # On Windows PowerShell, use 'Get-Command python'
```
The code is stable with python 3.6 and higher. The required python packages are listed in the requirements.txt file. To install the required packages, run the following command:

```
pip install -r requirements.txt
```

To deactivate the virtual environment after running the project, run the following command:

```
deactivate
```

### Step 3: Generate Embeddings
The [`run_embeddings.py`](/code/generate_embeddings/run_embeddings.py) script uses the RELISH Tokenized npy file as input and includes a default parameter dictionary with preset hyperparameters. You can easily adapt it for different values and parameters by modifying the `params_dict`. Make sure to have the RELISH Tokenized.npy file within the directory under the data folder.

To run this script, please execute the following command:

```
python3 code/generate_embeddings/run_embeddings.py --input "data/RELISH_tokenized.npy"
```

The script will create Doc2Vec models, generate embeddings, and store them in separate directories. You should expect to find a total of 18 files corresponding to the various models, embeddings, and embedding pickle files.

### Step 4: Calculate Cosine Similarity
In order to generate the cosine similarity matrix and execute this [script](/code/evaluation/generate_cosine_existing_pairs.py), run the following command:

```
python3 code/evaluation/generate_cosine_existing_pairs.py [-i INPUT] [-e EMBEDDINGS] [-o OUTPUT] [-c CORPUS]
```

You must pass the following four arguments:

+ -i/ --input : File path to the RELISH relevance matrix in the TSV format.
+ -e/ --embeddings : File path to the embeddings in the pickle file format.
+ -o/ --output : File path for the output 4 column cosine similarity matrix.
+ -c / --corpus : Name of the corpus (RELISH).


For example, if you are running the code from the code folder and have the RELISH relevance matrix in the data folder, run the cosine matrix creation for the first hyperparameter as:

```
python3 code/evaluation/generate_cosine_existing_pairs.py -i data/RELISH/Relevance_Matrix/RELISH.tsv -e dataframe/embeddings_pickle_0.pkl -o data/cosine_similarity_0.tsv -c RELISH
```


### Step 5: Hyperparameter Optimization

**_To be written_**

### Step 6: Precision@N
In order to calculate the Precision@N scores and execute this [script](/code/evaluation/precision.py), run the following command:

```
python3 code/evaluation/precision.py [-c COSINE FILE PATH]  [-o OUTPUT PATH]
```

You must pass the following two arguments:

+ -c/ --cosine_file_path: path to the 4-column cosine similarity existing pairs RELISH file: (tsv file)
+ -o/ --output_path: path to save the generated precision matrix: (tsv file)

For example, if you are running the code from the code folder and have the cosine similarity TSV file in the data folder, run the precision matrix creation for the first hyperparameter as:

```
python3 code/evaluation/precision.py -c data/cosine_similarity_0.tsv -o data/precision_doc2vec_0.tsv
```


### Step 7: nDCG@N
In order to calculate nDCG scores and execute this [script](/code/evaluation/calculate_gain.py), run the following command:

```
python3 code/evaluation/calculate_gain.py [-i INPUT]  [-o OUTPUT]
```

You must pass the following two arguments:

+ -i / --input: Path to the 4 column cosine similarity existing pairs RELISH TSV file.
+ -o/ --output: Output path along with the name of the file to save the generated nDCG@N TSV file.

For example, if you are running the code from the code folder and have the 4 column RELISH TSV file in the data folder, run the matrix creation for the first hyperparameter as:

```
python3 code/evaluation/calculate_gain.py -i data/cosine_similarity_0.tsv -o data/ndcg_doc2vec_0.tsv
```

### Step 8: Compile Results

In order to compile the average result values for Precison@ and nDCG@N and generate a single TSV file each, please use this [script](code/evaluation/show_avg.py).

You must pass the following two arguments:

+ -i / --input: Path to the directory consisting of all the precision matrices/gain matrices.
+ -o/ --output: Output path along with the name of the file to save the generated compiled Precision@N / nDCG@N TSV file.


If you are running the code from the code folder, run the compilation script as:

```
python3 code/evaluation/show_avg.py -i data/output/gain_matrices/ -o data/output/results_gain.tsv
```

NOTE: Please do not forget to put a `'/'` at the end of the input file path.

## Tutorial
A [tutorial](./docs/embeddings/) is accessible in the form of Jupyter notebook for the generation of embeddings.

