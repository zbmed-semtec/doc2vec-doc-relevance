import hyperparameter_optimization as hp
import embeddings as em
import update_matrix as um
from distribution_analysis import counting_table as ct
from distribution_analysis import ROC_curve as rc

# Generate hyperparameter combinations and save to tsv file (Uncomment if tsv file not found)
hp_df = hp.generate_hyperparameters(hp.params_d2v)
hp.save_file(hp_df, "Data/Hyperparameter/relish_hyperparameter_combinations.tsv")
print ("Hyperparameter Combinations Generated", flush=True)

# Retrieves cleaned data from .npy file 
pmids, titles, abstracts, docs = em.process_data_from_npy("Data/RELISH/TSV/RELISH_documents_pruned.npy")
print("Retrieved Cleaned Data", flush=True)

# Loop through each row in the hyper parameter tsv file
for index, row in hp_df.iterrows():
    print("Iterating Row: {}".format(index), flush=True)

    # Get hyper-parameters for the first row with AUC value = 0
    params = hp.get_parameters("Data/Hyperparameter/relish_hyperparameter_combinations.tsv")
    print (params, flush=True)
    print ("Retrieved Hyperparameter", flush=True)

    # Create and train Doc2Vec model
    model = em.createDoc2VecModel(pmids, docs, params)
    print ("Doc2Vec Model Generated", flush=True)

    # Update Relevance Matrix
    updated_matrix_file = "Data/Hyperparameter/Cosine_Similarities/relish_cosine_" + str(index) + ".tsv"
    um.update_relevance_matrix("Data/RELISH/Relevance_Matrix/RELISH.tsv", 
                            model, 
                            updated_matrix_file, 
                            "RELISH")
    print ("Relevance Matrix Updated", flush=True)

    # Load Relevance Matrix
    data = ct.load_relevance_matrix(updated_matrix_file)
    print ("Updated Relevance Matrix Loaded", flush=True)

    # Generate the counting table for the hyperparameter optimization process
    counting_table = ct.hp_create_counting_table(data, "RELISH", False) # True for TREC-repurposed, False otherwise
    print ("Counting table generated", flush=True)

    # Generate TPR and FPR values from the counting table required to plot the ROC curve
    counting_table = rc.generate_roc_values(counting_table, "RELISH", False) # True for TREC-repurposed, False otherwise
    print ("TPR and FPR values calculated", flush=True)

    # Calculate area under the ROC curve
    AUC_value = rc.calculate_auc(counting_table)
    print (AUC_value, flush=True)
    print ("AUC value calculated", flush=True)

    # Update AUC value in the hyper-parameter tsv file
    hp.update_file("Data/Hyperparameter/relish_hyperparameter_combinations.tsv", round(AUC_value, 4))
    print ("AUC value updated in the tsv file", flush=True)

print ("Completed")