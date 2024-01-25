import pandas as pd
def remove_rows_na(dataset: pd.DataFrame ):
    dataset["labels"] = dataset["labels"].astype('category')

    dataset['labels'] = dataset['labels'].cat.codes
    dataset = dataset.loc[dataset["labels"] != -1]
    return dataset

def remove_empty_rows(dataset: pd.DataFrame , column_name: str):
    return dataset[dataset[column_name].str.len() > 1]
