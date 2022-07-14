import generate_counting_table as ct

input_path = "Data/TREC/Relevance_Matrix/trec_relevance_matrix.tsv"

# Comment out this step if table exists with cosine similarities
data = ct.load_relevance_matrix(input_path)

data = ct.random_cosine_similarities(data)

counting_table = ct.create_counting_table(data)

print(counting_table)

output_path = "Playground/evaluation/TREC/counting_table.tsv"
ct.save_table(counting_table, output_path)

plot_path = "Playground/evaluation/TREC/output_plot.png"
ct.plot_graph(output_path, plot_path)

ct.create_counting_table_by_topic(data)