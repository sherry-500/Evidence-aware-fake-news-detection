import pandas as pd
import os
import json
import argparse

def align_datasets(df1, df2):
    """
    Arguments:
        df1 {pd.DataFrame}: the dataframe of Snopes
        df2 {pd.Dataframe}: the dataframe of PolitiFact
    Returns:
        
    """
    # rearrange the order of columns
    # original order:
    #     df1: <cred_label> <claim_source> <claim> <evidence> <evidence_source> 
    #     df2: <cred_label> <claim_id> <claim> <claim_source> <evidence> <evidence_source> 
    # rearranged order:
    #     <cred_label> <claim> <claim_source> <evidence> <evidence_source> 
    df1 = df1[[0, 2, 1, 3, 4]]
    df2 = df2[[0, 2, 3, 4, 5]]
    df1.columns = ['cred_label', 'claim', 'claim_source', 'evidence', 'evidence_source']
    df2.columns = ['cred_label', 'claim', 'claim_source', 'evidence', 'evidence_source']

    df1['cred_label'] = df1['cred_label'].map({
        'true': 'true',
        'mostly true': 'true',
        'false': 'false',
        'mostly false': 'false'
    })

    df2['cred_label'] = df2['cred_label'].map({
        'True': 'true', 
        'Half-True': 'true', 
        'Mostly True': 'true',
        'False': 'false', 
        'Pants on Fire!': 'false',
        'Mostly False': 'false'
    })

    # limit evidence num of each claim less than 29
    df2['rank'] = df2.groupby(['cred_label', 'claim', 'claim_source'])['evidence'].rank().astype('int')
    df2 = df2[df2['rank'] < 29].drop(columns='rank')
        
    return df1, df2

def df2json(df, file_path):
    """
    Group the dataframe by the column "claim" to create json formatted examples, and store the result in file 
    The json format: 
        {
            'cred_label': ,
            'claim': ,
            'claim_source': ,
            'evidences': [
                {'evidence': , 'evidence_source': },
                ...
            ]
        }
    Arguments:
        df {pd.Dataframe}: the dataframe originated from .tsv file
        file_path {string}: the file path where the result reserved
    """
    if os.path.exists(file_path):
        return

    f = open(file_path, 'w')
    df['cred_label'] = df['cred_label'].astype('str')
    grouped = df.groupby(['cred_label', 'claim_text', 'claim_source'])
    for (key, group) in grouped:
        group_dict = {
            'cred_label': key[0],
            'claim': key[1],
            'claim_source': key[2],
            'evidences': group[['evidence', 'evidence_source']].to_dict(orient='records')
        }
        group_json = json.dumps(group_dict)
        f.write(group_json + '\n')
    f.close()