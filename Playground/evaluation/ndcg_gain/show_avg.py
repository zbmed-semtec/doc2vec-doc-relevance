import pandas as pd
from os import listdir
from os.path import isfile, join

trec_path = "Data/TREC/nDCG-gain/nDCG/"
all_files = [f for f in listdir(trec_path) if isfile(join(trec_path, f))]

for file in all_files:
    file_path = trec_path + file
    df = pd.read_csv(file_path, sep='\t')
    print (file)
    print (df.tail(1))