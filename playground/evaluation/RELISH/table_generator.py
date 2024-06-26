import generate_counting_table as ct

input_path = "Data/RELISH/Relevance_Matrix/relish_relevance_matrix.csv"
data = ct.load_relevance_matrix(input_path)

data = ct.random_cosine_similarities(data)

counting_table = ct.create_counting_table(data)

print(counting_table)

output_path = "Playground/evaluation/RELISH/counting_table.tsv"
ct.save_table(counting_table, output_path)

plot_path = "Playground/evaluation/RELISH/output_plot.png"
ct.plot_graph(output_path, plot_path)