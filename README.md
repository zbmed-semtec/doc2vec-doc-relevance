# Doc2Vec-Doc-relevance
This repository focuses on an approach exploring and assessing literature-based doc-2-doc recommendations using the Doc2Vec technique with its application to the RELISH dataset. 

# Approach

 Our approach involves employing the [doc2vec](https://arxiv.org/pdf/1405.4053v2.pdf) model, which extends the popular word2vec technique to capture document-level semantics. By encoding documents and their textual content into fixed-length vectors, doc2vec facilitates similarity calculations and enables meaningful comparisons between documents. This approach is harnessed to derive insightful doc-2-doc recommendations within the realm of biomedical research, specificallY employing the RELISH dataset. In order to do so, we employ the [doc2vec model](https://radimrehurek.com/gensim/models/doc2vec.html) from the [Gensim](https://radimrehurek.com/gensim/index.html) library.

# Input Data

The input data for this method consists of preprocessed tokens derived from the RELISH documents. These tokens are stored in the RELISH.npy file, which contains preprocessed arrays comprising PMIDs, document titles, and abstracts. These arrays are generated through an extensive preprocessing pipeline, as elaborated in the [relish-preprocessing repository](https://github.com/zbmed-semtec/relish-preprocessing). Within this preprocessing pipeline, both the title and abstract texts undergo several stages of refinement: structural words are eliminated, text is converted to lowercase, and finally, tokenization is employed, resulting in arrays of individual words.


# Generating Embeddings
The following section outlines the process of generating document-level embeddings for each PMID of the RELISH corpus.

## Create Tagged Documents 
In this initial step, we create  `TaggedDocuments `, which associate each PMID with a corresponding list of words. Here, we combine the abstract and title of each document into a unified paragraph (or document). This unified text serves as the input for our Doc2Vec model, allowing it to capture the semantic meaning of the entire document.
## Generate Doc2Vec models 
In the second phase, we construct Doc2Vec models with customizable hyperparameters. These models are designed to understand the relationships between documents and words in a high-dimensional vector space. We employ the parameters shown below in order to generate our models.
#### Parameters

+ **dm:** {1,0} Refers to the training algorithm. If dm=1, distributed memory is used otherwise, distributed bag of words is used.
+ **vector_size:** It represents the number of dimensions our embeddings will have.
+ **window:** It represents the maximum distance between the current and predicted word.
+ **epochs:** It is the nuber of iterations of the training dataset.
+ **min_count:** It is the minimum number of appearances a word must have to not be ignored by the algorithm.

## Train the models
Training is a critical phase where our Doc2Vec models learn from the TaggedDocuments created in step one. The models adapt their internal weights through iterations to better represent document-level semantics.

## Extract embeddings
After model training, we can extract document-level embeddings. These embeddings are numerical vectors that represent the content and context of each document in a continuous vector space. These embeddings are stored by the model, associated with each PMID. For further downstream document similarity calculations we store the embeddings for each document with its PMID as a dataframe in a pickle file. A pickle file is created to store the embeddings for each specific set of hyperparameter combinations.

# Hyperparameter Optimization
*To be written*


# Code Implementation
The [`run_embeddings.py`](./code/generate_embeddings/run_embeddings.py) serves as a comprehensive wrapper function, supporting the creation of tagged documents, model generation, training, embedding generation, and the subsequent storage of these embeddings as pickle files. Individual functions for each task are provided in the other two code scripts:

+ [`embeddings.py`](./code/generate_embeddings/embeddings.py) : Creation of tagged documents from input tokens, creation and training of Doc2Vec models, generation of embeddings. 
+ [`embeddings.dataframe.py`](./code/generate_embeddings/embeddings_dataframe.py) : Creates a dataframe of embeddings with its corresponding PMID, sorts and stores it as a pickle file.


# Code Execution

The `run_embeddngs.py` script uses the RELISH Tokenized npy file as input and includes a default parameter dictionary with preset hyperparameters. You can easily adapt it for different values and parameters by modifying the `params_dict`.

To run this script, please execute the following command:

` python3 code/run_embeddings.py --input "data/RELISH_tokenized.npy"`

The script will generate models, embeddings, and pickle files, storing them in separately created directories.

# Tutorial
A [tutorial](./docs/embeddings/) is accessible in the form of Jupyter notebook for the generation of embeddings.
