# Evaluation of the doc2vec-doc-relevance using nDCG@N approach

The following tables show the results of the [nDCG@N](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Evaluation) evaluation approach, when applied on to "doc2vec-doc-relevance" technique. 
These results are calculated for the different hyper-parameters settings of the Doc2Vec approach to obtain the optimal combination for each dataset used in this work.

Each of the below tables contains seven columns. The first six of these columns represent the hyper-parameters of the Doc2Vec model. The remaining six columns represent the average nDCG scores at different values of N:
- **dm:** Defines the training algorithm that is used. If dm=0, 'distributed bag of words' (PV-DBOW) is used, else if dm=1, 'distributed memory' (PV-DM) is used.
- **epochs:** Defines the number of iterations over the corpus.
- **min_count:** Ignores those words that have the total frequency less than this number.
- **vector_size:** Defines the dimensionality of the feature vector.
- **window:** Defines the maximum distance between the current and predicted word within a sentence.
- **workers:** Defines the working threads used to train the model.
- **nDCG@5 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 5 articles retrieved.
- **nDCG@10 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 10 articles retrieved.
- **nDCG@15 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 15 articles retrieved.
- **nDCG@20 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 20 articles retrieved.
- **nDCG@25 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 25 articles retrieved.
- **nDCG@50 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 50 articles retrieved.

**RELISH:** The table below shows the results from the nDCG@N approach for the RELISH dataset.

| dm  | epochs  | min_count  | vector_size | window  | workers | nDCG@5 (AVG) | nDCG@10 (AVG) | nDCG@15 (AVG) | nDCG@20 (AVG) | nDCG@25 (AVG) | nDCG@50 (AVG) |
|:---:|:-------:|:----------:|:-----------:|:-------:|:-------:|:------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| 0   | 15      | 5          | 200         | 5       | 8       | 0.6606       | 0.6468        | 0.6504        |	0.6617        |	0.6767        |	0.7882        |
| 0   | 15      | 5          | 200         | 6       | 8       | 0.6608       | 0.6464        |	0.6493        |	0.6604        |	0.6753        |	0.7876        |
| 0   | 15      | 5          | 200         | 7       | 8       | 0.6588       | 0.6456        |	0.6499        |	0.6606        |	0.6762        |	0.7876        |
| 0   | 15      | 5          | 300         | 5       | 8       | 0.6667       | 0.6509        |	0.6541        |	0.6651        |	0.6798        |	0.7904        |
| 0   | 15      | 5          | 300         | 6       | 8       | 0.6649       | 0.6497        |	0.6538        |	0.6643        |	0.6802        |	0.7903        |
| 0   | 15      | 5          | 300         | 7       | 8       | 0.6622       | 0.6498        |	0.6523        |	0.6637        |	0.6803        |	0.7898        |
| 0   | 15      | 5          | 400         | 5       | 8       | 0.6671       | 0.6507        |	0.6543        |	0.6657        |	0.6804        |	0.7909        |
| 0   | 15      | 5          | 400         | 6       | 8       | 0.6668       | 0.6527        |	0.6557        |	0.6656        |	0.6813        |	0.7919        |
| 0   | 15      | 5          | 400         | 7       | 8       | 0.6655       | 0.6499        |	0.6546        |	0.6652        |	0.6804        |	0.7906        |
| 1   | 15      | 5          | 200         | 5       | 8       | 0.6576       | 0.6462        |	0.6503        |	0.6597        |	0.6754        |	0.7869        |
| 1   | 15      | 5          | 200         | 6       | 8       | 0.6584       | 0.6459        |	0.6493        |	0.6595        |	0.6744        |	0.7867        |
| 1   | 15      | 5          | 200         | 7       | 8       | 0.6575       | 0.6455        |	0.6492        |	0.6597        |	0.6741        |	0.7864        |
| 1   | 15      | 5          | 300         | 5       | 8       | 0.6588       | 0.647         |	0.6512        |	0.6615        |	0.6767        |	0.7877        |
| 1   | 15      | 5          | 300         | 6       | 8       | 0.6591       | 0.6469        |	0.6496        |	0.6609        |	0.6759        |	0.787         |
| 1   | 15      | 5          | 300         | 7       | 8       | 0.6582       | 0.6472        |	0.65          |	0.6598        |	0.6755        |	0.7867        |
| 1   | 15      | 5          | 400         | 5       | 8       | 0.6597       | 0.648         |	0.65          |	0.6612        |	0.6761        |	0.7877        |
| 1   | 15      | 5          | 400         | 6       | 8       | 0.6576       | 0.6465        |	0.6484        |	0.6596        |	0.6749        |	0.7866        |
| 1   | 15      | 5          | 400         | 7       | 8       | 0.6561       | 0.6465        |	0.6486        |	0.6587        |	0.6747        |	0.7868        |

**TREC-repurposed:** The table below shows the results from the nDCG@N approach for the "TREC-repurposed" variant of the TREC dataset.


| dm  | epochs  | min_count  | vector_size | window  | workers | nDCG@5 (AVG) | nDCG@10 (AVG) | nDCG@15 (AVG) | nDCG@20 (AVG) | nDCG@25 (AVG) | nDCG@50 (AVG) |
|:---:|:-------:|:----------:|:-----------:|:-------:|:-------:|:------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| 0   | 15      | 5          | 200         | 5       | 8       | 0.4871       | 0.4806        | 0.4771        |	0.4769        |	0.4774        |	0.4891        |
| 0   | 15      | 5          | 200         | 6       | 8       | 0.4871       | 0.4792        |	0.4763        |	0.4765        |	0.4774        |	0.4893        |
| 0   | 15      | 5          | 200         | 7       | 8       | 0.4888       | 0.4808        |	0.4782        |	0.4775        |	0.4783        |	0.4891        |
| 0   | 15      | 5          | 300         | 5       | 8       | 0.4869       | 0.4812        |	0.4781        |	0.4778        |	0.4788        |	0.4906        |
| 0   | 15      | 5          | 300         | 6       | 8       | 0.4842       | 0.478         |	0.476         |	0.4751        |	0.4761        |	0.4878        |
| 0   | 15      | 5          | 300         | 7       | 8       | 0.4883       | 0.4807        |	0.479         |	0.4779        |	0.4789        |	0.4905        |
| 0   | 15      | 5          | 400         | 5       | 8       | 0.4879       | 0.4813        |	0.4777        |	0.4775        |	0.4788        |	0.4903        |
| 0   | 15      | 5          | 400         | 6       | 8       | 0.4873       | 0.4811        |	0.4782        |	0.4767        |	0.4777        |	0.49          |
| 0   | 15      | 5          | 400         | 7       | 8       | 0.4894       | 0.4817        |	0.4784        |	0.4786        |	0.4791        |	0.4902        |
| 1   | 15      | 5          | 200         | 5       | 8       | 0.4671       | 0.4607        |	0.4599        |	0.4599        |	0.4615        |	0.4735        |
| 1   | 15      | 5          | 200         | 6       | 8       | 0.4679       | 0.4612        |	0.4594        |	0.459         |	0.46          |	0.472         |
| 1   | 15      | 5          | 200         | 7       | 8       | 0.458        | 0.4514        |	0.4499        |	0.4503        |	0.4514        |	0.4645        |
| 1   | 15      | 5          | 300         | 5       | 8       | 0.4676       | 0.463         |	0.4602        |	0.4601        |	0.4612        |	0.4749        |
| 1   | 15      | 5          | 300         | 6       | 8       | 0.4608       | 0.457         |	0.4551        |	0.4558        |	0.4571        |	0.4701        |
| 1   | 15      | 5          | 300         | 7       | 8       | 0.4659       | 0.4601        |	0.4579        |	0.4572        |	0.4584        |	0.4697        |
| 1   | 15      | 5          | 400         | 5       | 8       | 0.466        | 0.4614        |	0.4598        |	0.4603        |	0.4619        |	0.4743        |
| 1   | 15      | 5          | 400         | 6       | 8       | 0.4626       | 0.4579        |	0.4565        |	0.4562        |	0.4573        |	0.4709        |
| 1   | 15      | 5          | 400         | 7       | 8       | 0.4551       | 0.4524        |	0.451         |	0.4513        |	0.4527        |	0.4666        |

