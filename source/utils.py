'''
utility functions
'''

__author__ = 'Oguzhan Gencoglu'

import os
from os.path import join
from os.path import abspath
import json

import pandas as pd
import numpy as np

from configs import config as cf


def is_available(filename):
    '''
    [filename] : str
    '''

    return os.path.isfile(filename)


def chunks(lst, n):
    '''
    Yield successive n-sized chunks from list
    [lst] : python list
    [n]   : int
    '''
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def read_intent_dataset(verbose=True):
    '''
    Load 'Intent' dataset
    [verbose] : bool, verbosity level
    '''
    # read as a pandas dataframe
    data = []
    for lang in ['en', 'es', 'fr']:
        for ds in ['train', 'test', 'eval']:
            path = abspath(join(cf.INTENT_DIR, lang, '{}.tsv'.format(ds)))
            df = pd.read_csv(path, header=None, sep='\t',
                             names=['text', 'class'])
            data.append(df)
    data = pd.concat(data)

    # merge certain categories (see configs.py) and rename columns
    data['class'] = data['class'].replace(cf.intent_label_map)

    # remove trivial (too easy) categories
    for cat in ['hi', 'okay_thanks']:
        data = data[data['class'] != 'intent:{}'.format(cat)]

    if verbose:
        print('\t"Intent" data shape={}'.format(data.shape))

    return data


def read_questions_dataset(verbose=True):
    '''
    Load 'Questions' dataset
    [verbose] : bool, verbosity level
    '''
    # read as a pandas dataframe
    data_path = abspath(join(cf.QUESTIONS_DIR, 'final_master_dataset.csv'))
    data = pd.read_csv(data_path, delimiter=',',
                       usecols=['Question', 'Category'])
    data.rename(columns={'Question': 'text', 'Category': 'class'},
                inplace=True)
    data = data[~data['class'].isna()]  # remove unannotated rows

    # split label into class and subclass, keep only class
    data[['class', 'subclass']] = data['class'].str.split('-', 1, expand=True)
    data['class'] = data['class'].str.strip()
    data.drop(['subclass'], axis=1, inplace=True)

    data = data[[i in cf.questions_relevant_categories for i in data['class']]]

    if verbose:
        print('\t"Questions" data shape={}'.format(data.shape))

    return data


def merge_datasets(embeddings='labse', verbose=True):
    '''
    Merge 'Intent' and 'Questions' datasets
    [embeddings] : str, type of embeddings to load ('bert' or 'labse')
    [verbose] : bool, verbosity level
    '''
    # load datasets
    intent = read_intent_dataset(verbose=False)
    questions = read_questions_dataset(verbose=False)
    merged = pd.concat([intent, questions])

    # load corresponding embeddings
    if embeddings == 'labse':
        emb_to_load = (cf.intent_embeddings, cf.questions_embeddings)
    elif embeddings == 'bert':
        emb_to_load = (cf.intent_embeddings_bert, cf.questions_embeddings_bert)
    else:
        raise ValueError("embeddings argument can be 'bert' or 'labse'")
    print(f'{embeddings} embeddings loaded.')

    intent_embeddings = np.load(abspath(join(cf.INTENT_DIR, emb_to_load[0])))
    questions_embeddings = np.load(abspath(join(cf.QUESTIONS_DIR,
                                                emb_to_load[1])))
    merged_embeddings = np.vstack([intent_embeddings, questions_embeddings])

    assert merged.shape[0] == merged_embeddings.shape[0]

    if verbose:
        print('Full data shape={}'.format(merged.shape))

    return merged, merged_embeddings


# _____________ Logging related functions _____________
def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def save_logs(logs_dict, dict_name):
    '''
    Save best hyperparameters dictionary to "logs" directory
    [logs_dict] : dict
    [dict_name] : str
    '''

    json.dump(logs_dict,
              open('{}/{}.json'.format(cf.LOGS_DIR,
                                       dict_name),
                   'w'), default=convert)
    print('Best hyper-parameters saved...')

    return None


def load_logs(dict_name):
    '''
    Load best hyperparameters dictionary from "logs" directory
    [dict_name]   : str
    '''
    log_path = '{}/{}.json'.format(cf.LOGS_DIR, dict_name)

    if not is_available(log_path):
        raise ValueError('Hyperparameters are not available. '
                         'Please run train.py in "hyper_opt" mode before full '
                         'training.')

    with open() as logs_json:
        logs = json.load(logs_json)
    print('Best hyperparameters loaded...')

    return logs
