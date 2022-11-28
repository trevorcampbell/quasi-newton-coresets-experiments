import hashlib
import os
import json
import pickle as pk
import pandas as pd
import numpy as np

def find_matching(to_match, results_folder = 'results/', log_file = 'manifest.csv'):
    # immediately return no match if there's no results folder
    if not os.path.exists(os.path.join(results_folder, log_file)):
        return []
    # load the manifest
    with open(os.path.join(results_folder, log_file), 'r') as f:
        manifest = f.readlines()

    # remove the 'func' argument (cant hash function objects)
    to_match = to_match.copy() # avoid editing dict without caller knowing
    to_match.pop('func', None)

    # split each manifest line into [hash, args_string]
    manifest = [ line.split(':', 1) for line in manifest]
    # find matching manifest lines
    matching_hashes = []
    for line in manifest:
        str_args = line[1].strip()
        args_dict = {key : val for (key, val) in json.loads(str_args).items() if key in to_match}
        if to_match == args_dict:
            matching_hashes.append(line[0].strip())
    return matching_hashes

def check_exists(arguments, results_folder = 'results/', log_file = 'manifest.csv'):
    matching_hashes = find_matching(vars(arguments), results_folder, log_file)
    if len(matching_hashes) == 0:
        return None
    if len(matching_hashes) > 1:
        raise ValueError(f"ERROR: Multiple matching results cache files for arguments. Arguments: {arguments} Matching hashes: {matching_hashes}")
    if os.path.exists(os.path.join(results_folder, matching_hashes[0]+'.csv')):
        return os.path.join(results_folder, matching_hashes[0]+'.csv')
    return None

def load_matching(arguments, match = [], results_folder = 'results/', log_file = 'manifest.csv'):
    to_match = {key : val for (key,val) in vars(arguments).items() if key in match}
    print("Plot: Matching arguments setting {to_match}")
    matching_hashes = find_matching(to_match, results_folder, log_file)
    if len(matching_hashes) == 0:
        raise ValueError(f"ERROR: no matches for plotting. to_match = {to_match}")
    df = pd.DataFrame()
    for mash in matching_hashes:
        df_row = pd.read_csv(os.path.join(results_folder, mash+".csv"))
        df = df.append(df_row, ignore_index=True)
    print("Plotting data in dataframe:")
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(df)
    return df

def save(arguments, results_folder = 'results/', log_file = 'manifest.csv', **kwargs):

    # convert the arguments namespace to a dictionary
    nsdict = vars(arguments)

    # remove the 'func' argument (cant hash function objects)
    nsdict.pop('func', None)

    # hash the input arguments
    arg_hash = hashlib.md5(json.dumps(nsdict, sort_keys=True).encode('utf-8')).hexdigest()

    #make the results folder if it doesn't exist
    if not os.path.exists(results_folder):
      os.mkdir(results_folder)

    # if the file doesn't already exist, create the df file and append a line to manifest
    if not os.path.exists(os.path.join(results_folder, arg_hash+'.csv')):
        with open(os.path.join(results_folder, log_file), 'a') as f:
            manifest_line = arg_hash+':'+ json.dumps(nsdict, sort_keys=True) + '\n'
            f.write(manifest_line)

    # add the kwargs into the data dict
    for key, val in kwargs.items():
        if key in nsdict:
            raise ValueError(f"ERROR: key {key} (val {val}) already in namespace; cannot save this as data. Namespace: {arguments}")
        if key != 'func':
            nsdict[key] = val

    #save the df, overwriting a previous result
    df = pd.DataFrame({key:[val] for (key,val) in nsdict.items()})
    df.to_csv(os.path.join(results_folder, arg_hash+'.csv'), index=False)

