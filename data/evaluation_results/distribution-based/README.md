# Evaluation of the doc2vec-doc-relevance with the distribution-based approach

The following tables show the results of the [distribution-based](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Distribution_Analysis) evaluation approach, when applied on to "doc2vec-doc-relevance" technique. 
These results are calculated for the different hyper-parameters settings of the Doc2Vec approach to obtain the optimal combination for each dataset used in this work.

Each of the below tables contains seven columns. The first six of these columns represent the hyper-parameters of the Doc2Vec model:
- **dm:** Defines the training algorithm that is used. If dm=0, 'distributed bag of words' (PV-DBOW) is used, else if dm=1, 'distributed memory' (PV-DM) is used.
- **epochs:** Defines the number of iterations over the corpus.
- **min_count:** Ignores those words that have the total frequency less than this number.
- **vector_size:** Defines the dimensionality of the feature vector.
- **window:** Defines the maximum distance between the current and predicted word within a sentence.
- **workers:** Defines the working threads used to train the model.
- **AUC:** Defines the "area under the curve", as a result of this evaluation approach.

**RELISH:** The table below shows the results from the distribution-based approach for the RELISH dataset.

| dm    | epochs      | min_count     | vector_size | window  | workers | AUC    |
|:-----:|:-----------:|:-------------:|:-----------:|:-------:|:-------:|:------:|
| 0     | 15          | 5             | 200         | 5       | 8       | 0.5825 |
| 0     | 15          | 5             | 200         | 6       | 8       | 0.5821 |
| 0     | 15          | 5             | 200         | 7       | 8       | 0.581  |
| 0     | 15          | 5             | 300         | 5       | 8       | 0.5826 |
| 0     | 15          | 5             | 300         | 6       | 8       | 0.5828 |
| 0     | 15          | 5             | 300         | 7       | 8       | 0.5819 |
| 0     | 15          | 5             | 400         | 5       | 8       | 0.5828 |
| 0     | 15          | 5             | 400         | 6       | 8       | 0.5816 |
| 0     | 15          | 5             | 400         | 7       | 8       | 0.5824 |
| 1     | 15          | 5             | 200         | 5       | 8       | 0.596  |
| 1     | 15          | 5             | 200         | 6       | 8       | 0.5946 |
| 1     | 15          | 5             | 200         | 7       | 8       | 0.5955 |
| 1     | 15          | 5             | 300         | 5       | 8       | 0.5947 |
| 1     | 15          | 5             | 300         | 6       | 8       | 0.5954 |
| 1     | 15          | 5             | 300         | 7       | 8       | 0.5931 |
| 1     | 15          | 5             | 400         | 5       | 8       | 0.594  |
| 1     | 15          | 5             | 400         | 6       | 8       | 0.5939 |
| 1     | 15          | 5             | 400         | 7       | 8       | 0.5913 |

**TREC-simplified:** The table below shows the results from the distribution-based approach for the "TREC-simplified" variant of the TREC dataset.

| dm    | epochs      | min_count     | vector_size | window  | workers | AUC    |
|:-----:|:-----------:|:-------------:|:-----------:|:-------:|:-------:|:------:|
| 0     | 15          | 5             | 200         | 5       | 8       | 0.6518 |
| 0     | 15          | 5             | 200         | 6       | 8       | 0.6521 |
| 0     | 15          | 5             | 200         | 7       | 8       | 0.6519 |
| 0     | 15          | 5             | 300         | 5       | 8       | 0.6513 |
| 0     | 15          | 5             | 300         | 6       | 8       | 0.6516 |
| 0     | 15          | 5             | 300         | 7       | 8       | 0.6515 |
| 0     | 15          | 5             | 400         | 5       | 8       | 0.6516 |
| 0     | 15          | 5             | 400         | 6       | 8       | 0.6514 |
| 0     | 15          | 5             | 400         | 7       | 8       | 0.6511 |
| 1     | 15          | 5             | 200         | 5       | 8       | 0.6415 |
| 1     | 15          | 5             | 200         | 6       | 8       | 0.6402 |
| 1     | 15          | 5             | 200         | 7       | 8       | 0.6387 |
| 1     | 15          | 5             | 300         | 5       | 8       | 0.6412 |
| 1     | 15          | 5             | 300         | 6       | 8       | 0.6408 |
| 1     | 15          | 5             | 300         | 7       | 8       | 0.6377 |
| 1     | 15          | 5             | 400         | 5       | 8       | 0.6408 |
| 1     | 15          | 5             | 400         | 6       | 8       | 0.6392 |
| 1     | 15          | 5             | 400         | 7       | 8       | 0.637  |

**TREC-repurposed:** The table below shows the results from the distribution-based approach for the "TREC-repurposed" variant of the TREC dataset.

| dm    | epochs      | min_count     | vector_size | window  | workers | AUC    |
|:-----:|:-----------:|:-------------:|:-----------:|:-------:|:-------:|:------:|
| 0     | 15          | 5             | 200         | 5       | 8       | 0.7697 |
| 0     | 15          | 5             | 200         | 6       | 8       | 0.77   |
| 0     | 15          | 5             | 200         | 7       | 8       | 0.7698 |
| 0     | 15          | 5             | 300         | 5       | 8       | 0.7699 |
| 0     | 15          | 5             | 300         | 6       | 8       | 0.7694 |
| 0     | 15          | 5             | 300         | 7       | 8       | 0.7703 |
| 0     | 15          | 5             | 400         | 5       | 8       | 0.77   |
| 0     | 15          | 5             | 400         | 6       | 8       | 0.7699 |
| 0     | 15          | 5             | 400         | 7       | 8       | 0.7699 |
| 1     | 15          | 5             | 200         | 5       | 8       | 0.7481 |
| 1     | 15          | 5             | 200         | 6       | 8       | 0.7474 |
| 1     | 15          | 5             | 200         | 7       | 8       | 0.746  |
| 1     | 15          | 5             | 300         | 5       | 8       | 0.7488 |
| 1     | 15          | 5             | 300         | 6       | 8       | 0.7452 |
| 1     | 15          | 5             | 300         | 7       | 8       | 0.7456 |
| 1     | 15          | 5             | 400         | 5       | 8       | 0.7498 |
| 1     | 15          | 5             | 400         | 6       | 8       | 0.7447 |
| 1     | 15          | 5             | 400         | 7       | 8       | 0.7458 |

