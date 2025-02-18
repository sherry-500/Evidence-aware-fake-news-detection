import pandas as pd
import os
import json
import argparse

def align_datasets(df1, df2):
    """
    Arguments:
        df {pd.DataFrame}: the dataframe of Snopes/PolitiFact
    Returns:
        
    """
    # rearrange the order of columns
    # original order:
    #     snope_df: <cred_label> <claim_source> <claim> <evidence> <evidence_source> 
    #     politifact_df: <cred_label> <claim_id> <claim> <claim_source> <evidence> <evidence_source> 
    # rearranged order:
    #     <cred_label> <claim> <claim_source> <evidence> <evidence_source> 
    if df.shape[1] == 5:
        # Snopes
        df = df[[0, 2, 1, 3, 4]]
    else:
        # PolitiFact
        df = df[[0, 2, 3, 4, 5]]
    df.columns = ['cred_label', 'claim', 'claim_source', 'evidence', 'evidence_source']
    return df

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
    grouped = df.groupby(['cred_label', 'claim', 'claim_source'])
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile')
    parser.add_argument('--outfile')
    args = parser.parse_args()
    
    df = pd.read_csv(args.infile, sep='\t', header=None)
    df = align_datasets(df)
    df2json(df, args.outfile)