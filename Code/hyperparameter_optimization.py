import pandas as pd

def generate_hyperparameters(params: dict) -> pd.DataFrame:
    """
    Generate combinations of hyperparameters.

    Parameters
    ----------
    params: dict
        Dictionary containing the Doc2Vec parameters.
    Returns
    -------
    df_hp: pd.DataFrame
        Pandas Dataframe containing all the combinations 
        of hyperparameters.
    """
    try:
        from sklearn.model_selection import ParameterGrid

        param_grid = ParameterGrid(params)
        df_hp = pd.DataFrame.from_dict(param_grid)

        return df_hp
    except:
        import itertools

        keys, values = zip(*params.items())
        df_hp = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return df_hp

def save_file(hp_df: pd.DataFrame, output_path: str) -> None:
    """
    Add the AUC (area under the curve) to the pandas 
    dataframe and save all the combinations along with 
    the AUC column to a tsv file.

    Parameters
    ----------
    hp_df: pd.DataFrame
        Pandas Dataframe containing all the combinations 
        of hyperparameters.
    output_path: str
        File path where the tsv file will be saved.
    """
    hp_df['AUC'] = 0
    hp_df.to_csv(output_path, sep="\t", index=False)

def get_parameters(input_file: str, index: int) -> dict:
    """
    Retrieve the set of parameters at a particular index 
    to be used in the Doc2Vec model.

    Parameters
    ----------
    input_file: str
        TSV file containing all the combinations of the 
        hyperparameters.
    index: int
        Index from which the Doc2Vec parameters are to 
        be retrieved.
    """
    hp_df = pd.read_csv(input_file, delimiter="\t")
    df = hp_df.iloc[index]
    params = df.to_dict()
    
    # Remove AUC from the dictionary
    del params["AUC"]

    return params

def update_file(file_path: str, index: int, AUC_value: float) -> None:
    """
    Updates the AUC value in the tsv file at the specified index.

    Parameters
    ----------
    file_path: str
        File path of the tsv file which will be updated.
    index: int
        Index at which the AUC value will be updated.
    AUC_value: float
        AUC value which will be stored.
    """
    hp_df = pd.read_csv(file_path, delimiter="\t")
    hp_df.at[index, "AUC"] = AUC_value
    hp_df.to_csv(file_path, sep="\t", index=False)

# Results in 18 combinations
params_d2v = {
    "dm": [0, 1],
    "vector_size": [200, 300, 400], 
    "window": [5, 6, 7], 
    "min_count": [1], 
    "epochs": [5], 
    "workers": [8]}

