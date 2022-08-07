import hyperparameter_optimization as hp
import embeddings as em
import update_matrix as um
from distribution_analysis import counting_table as ct
from distribution_analysis import ROC_curve as rc

# Generate hyperparameter combinations and save to tsv file (Uncomment if tsv file not found)
# hp_df = hp.generate_hyperparameters(hp.params_d2v)
# hp.save_file(hp_df, "Data/Hyperparameter/hyperparameter_combinations.tsv")

# Get hyper-parameters for the first row with AUC value = 0
params = hp.get_parameters("Data/Hyperparameter/hyperparameter_combinations.tsv")
print ("Retrieved Hyperparameter")

# Retrieves cleaned data from .npy file 
pmids, titles, abstracts, docs = em.process_data_from_npy("Data/RELISH/TSV/RELISH_documents_pruned.npy")
print("Retrieved Cleaned Data")

# Create and train Doc2Vec model
model = em.createDoc2VecModel(pmids, docs, params)
print ("Doc2Vec Model Generated")

# Update Relevance Matrix
um.update_relevance_matrix("Data/RELISH/Relevance_Matrix/relish_relevance_matrix.tsv", 
                           model, 
                           "Data/RELISH/Relevance_Matrix/relish_relevance_matrix_updated.tsv", 
                           "RELISH")
print ("Relevance Matrix Updated")

# Load Relevance Matrix
data = ct.load_relevance_matrix("Data/RELISH/Relevance_Matrix/relish_relevance_matrix_updated.tsv")
print ("Updated Relevance Matrix Loaded")

# Generate the counting table for the hyperparameter optimization process
counting_table = ct.hp_create_counting_table(data, "RELISH", False)
print ("Counting table generated")

# Save the counting table
ct.save_table(counting_table, "Data/RELISH/Relevance_Matrix/relish_counting_table.tsv")
print ("Counting table saved")

# Generate TPR and FPR values from the counting table required to plot the ROC curve
counting_table = rc.generate_roc_values(counting_table, "RELISH", False)
print ("TPR and FPR values calculated")

# Calculate area under the ROC curve
AUC_value = rc.calculate_auc(counting_table)
print ("AUC value calculated")

# Update AUC value in the hyper-parameter tsv file
hp.update_file("Data/Hyperparameter/hyperparameter_combinations.tsv", round(AUC_value, 2))
print ("AUC value updated in the tsv file")