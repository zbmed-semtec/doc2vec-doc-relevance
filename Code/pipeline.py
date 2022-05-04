import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

en_stopwords = set(stopwords.words('english'))
irrelevant_characters = [":", ";", "\'", "\"", "[", "]"]

lemmatizer = WordNetLemmatizer()

# Convert into lowercase characters
def convert_lowercase(text):
    return str(text).lower()

# Remove apostrophes from the text
def remove_apostrophes(text):
    text = text.replace("'", "")
    text = text.replace('"', "")
    return text

# Tokenization
def tokenization(text):
    return word_tokenize(text)

# Remove stopwords
def remove_stopwords(token):
    return [item for item in token if item not in en_stopwords]

# Remove irrelevant characters
def remove_irrelevant_characters(token):
    return [item for item in token if item not in irrelevant_characters]

# Get POS tags for lemmatization
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def get_pos_tag(word):
    word_tag = pos_tag([word])
    return get_wordnet_pos(word_tag[0][1])

# Lemmatization
def lemmatization(token):
    return [lemmatizer.lemmatize(word=w,pos=get_pos_tag(w)) for w in token]

# Detokenize the tokens and convert them back to string
def detokenize(token):
    return TreebankWordDetokenizer().detokenize(token)

# Remove whitespaces before but the punctuations
def remove_whitespaces_before_punctuations(text):
    return re.sub(r'\s([?.!,"](?:\s|$))', r'\1', text)

def process_data_from_tsv(filePathIn=None):
    pmids = []
    titles = []
    abstracts = []

    corpus = pd.read_csv(filePathIn, sep='\t')

    for index, row in corpus.iterrows():
        # Converting title and abstract into lowercase
        row['title'] = convert_lowercase(row['title'])
        row['abstract'] = convert_lowercase(row['abstract'])

        # Remove apostrophes from the title and abstract
        row['title'] = remove_apostrophes(row['title'])
        row['abstract'] = remove_apostrophes(row['abstract'])

        # Tokenizing the title and abstract
        row['title'] = tokenization(row['title'])
        row['abstract'] = tokenization(row['abstract'])

        # Remove stopwords from the title and abstract
        row['title'] = remove_stopwords(row['title'])
        row['abstract'] = remove_stopwords(row['abstract'])

        # Remove irrelevant characters from the title and abstract
        row['title'] = remove_irrelevant_characters(row['title'])
        row['abstract'] = remove_irrelevant_characters(row['abstract'])

        # Lemmatize the tokens in the title and abstract
        row['title'] = lemmatization(row['title'])
        row['abstract'] = lemmatization(row['abstract'])

        # Detokenize the tokens in the title and abstract
        row['title'] = detokenize(row['title'])
        row['abstract'] = detokenize(row['abstract'])

        # Remove whitespaces before punctuations in the title and abstract
        row['title'] = remove_whitespaces_before_punctuations(row['title'])
        row['abstract'] = remove_whitespaces_before_punctuations(row['abstract'])

        corpus.at[index,'title'] = row['title']
        corpus.at[index,'abstract'] = row['abstract']

        pmids.append(row['PMID'])

        title = []
        title.append(row['title'])
        titles.append(title)

        abstract = []
        abstract.append(row['abstract'])
        abstracts.append(abstract)

    return (pmids, titles, abstracts)

my_list = process_data_from_tsv('Data/TREC/TSV/sample.tsv')
print (my_list[1])