# Create model and embeddings using the optimized hyperparameter settings

import embeddings as em
import update_matrix as um

##################### RELISH #####################################################################################

# Retrieves cleaned data from .npy file 
pmids, titles, abstracts, docs = em.process_data_from_npy("Data/RELISH/TSV/RELISH_documents_pruned.npy")
print("Retrieved RELISH Cleaned Data", flush=True)

params = {'dm': 0, 'epochs': 5, 'min_count': 1, 'vector_size': 300, 'window': 7, 'workers': 8}

# Create and train Doc2Vec model
model = em.createDoc2VecModel(pmids, docs, params)
print ("RELISH Doc2Vec Model Generated", flush=True)

# Save the model generated
em.saveDoc2VecModel(model, 'Models/RELISH/relish_doc2vec.model')
print ("RELISH Doc2Vec Model Saved", flush=True)

# Generate the embeddings
em.create_document_embeddings(pmids, model, 'Embeddings/RELISH/')
print ("RELISH Embeddings Generated", flush=True)

# Update Relevance Matrix
um.update_relevance_matrix("Data/RELISH/Relevance_Matrix/RELISH.tsv", 
                        model, 
                        "Data/RELISH/Relevance_Matrix/relish_relevance_matrix_cosine.tsv", 
                        "RELISH")
print ("RELISH Relevance Matrix Updated", flush=True)

##################################### TREC ###################################################################################

# Retrieves cleaned data from .npy file 
pmids, titles, abstracts, docs = em.process_data_from_npy("Data/TREC/TSV/TREC_documents_pruned.npy")
print("Retrieved TREC Cleaned Data", flush=True)

params = {'dm': 0, 'epochs': 5, 'min_count': 1, 'vector_size': 200, 'window': 7, 'workers': 8}

# Create and train Doc2Vec model
model = em.createDoc2VecModel(pmids, docs, params)
print ("TREC Doc2Vec Model Generated", flush=True)

# Save the model generated
em.saveDoc2VecModel(model, 'Models/TREC/trec_doc2vec.model')
print ("TREC Doc2Vec Model Saved", flush=True)

# Generate the embeddings
em.create_document_embeddings(pmids, model, 'Embeddings/TREC/')
print ("TREC Embeddings Generated", flush=True)

# Update Relevance Matrix
um.update_relevance_matrix("Data/TREC/Relevance_Matrix/trec_simplified_relevance_matrix.tsv", 
                        model, 
                        "Data/TREC/Relevance_Matrix/trec_simplified_cosine.tsv", 
                        "TREC")
print ("TREC-simplified Relevance Matrix Updated", flush=True)

um.update_relevance_matrix("Data/TREC/Relevance_Matrix/trec_repurposed_matrix.tsv", 
                        model, 
                        "Data/TREC/Relevance_Matrix/trec_repurposed_cosine.tsv", 
                        "TREC")
print ("TREC-repurposed Relevance Matrix Updated", flush=True)
