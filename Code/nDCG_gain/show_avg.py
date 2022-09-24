# Show nDCG average for all the hyper-parameter settings

import re
import pandas as pd
from os import listdir
from os.path import isfile, join

# Path to nDCG folder
nDCG_path = "Data/RELISH/nDCG-gain/nDCG/"
# Get all files in the folder
all_files = [f for f in listdir(nDCG_path) if isfile(join(nDCG_path, f))]
# Sort files in numerical order
all_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

combined_list = []

# Output file path
output_path = 'Data/RELISH/nDCG-gain/relish_nDCG_gain_results.tsv'

# Loop through the files
for index, file in enumerate(all_files):
    file_path = nDCG_path + file
    # Read each tsv file
    df = pd.read_csv(file_path, sep='\t')
    # Remove the index column in the Pandas dataframe
    df = df.iloc[: , 1:]
    # Get only the last row (containing the average values)
    df = df.tail(1)

    # Convert the last row to list and append to 'combined_list'
    flattened = [val for sublist in df.values.tolist() for val in sublist]
    combined_list.append(flattened)

    # For the final loop, create a dataframe using 'combined_list' and write to tsv file
    if index == (len(all_files) - 1):
        combined_df = pd.DataFrame(combined_list, columns=df.columns.values.tolist())

        combined_df.to_csv(output_path, sep='\t')