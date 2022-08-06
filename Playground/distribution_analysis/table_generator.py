import counting_table as ct

############################ Input file path for RELISH relevance matrix with cosine similarities ############################################

input_path = "Data/RELISH/Relevance_Matrix/relish_relevance_matrix_updated.tsv"
plot_path = "Data/RELISH/Relevance_Matrix/relish_plot.png"
output_path = "Data/RELISH/Relevance_Matrix/relish_counting_table.tsv"

############################ Input file path for TREC-repurposed relevance matrix with cosine similarities ####################################

# input_path = "Data/TREC/Relevance_Matrix/trec_repurposed_matrix_updated.tsv"
# plot_path = "Data/RELISH/Relevance_Matrix/trec_repurposed_plot.png"
# output_path = "Data/TREC/Relevance_Matrix/trec_repurposed_counting_table.tsv"

############################ Input file path for TREC-simplified relevance matrix with cosine similarities ####################################

# input_path = "Data/TREC/Relevance_Matrix/trec_simplified_relevance_matrix_updated.tsv"
# plot_path = "Data/RELISH/Relevance_Matrix/trec_simplified_plot.png"
# output_path = "Data/TREC/Relevance_Matrix/trec_simplified_counting_table.tsv"


# Load the relevance matrix
data = ct.load_relevance_matrix(input_path)

# Generate the counting table
counting_table = ct.create_counting_table(data, "RELISH", False)

# Generate the counting table for the hyperparameter optimization process
counting_table = ct.hp_create_counting_table(data, "RELISH", False)

# Plot the graph
ct.plot_graph(counting_table, "RELISH", False, False, True, plot_path)

# Save table
ct.save_table(counting_table, output_path)
