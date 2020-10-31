'''
Script to extract BERT Embeddings
'''

__author__ = 'Oguzhan Gencoglu'

from os.path import join
from os.path import abspath
import argparse

import numpy as np
from tqdm import tqdm
from vectorhub.encoders.text.torch_transformers import Transformer2Vec

from utils import (is_available, read_intent_dataset, read_questions_dataset)
from configs import config as cf


def encode_bert(list_of_strings, model):
    '''
    [list_of_strings] : lsit of input strings to encode
    [model]           : BERT model
    '''

    return np.array([model.encode(i) for i in tqdm(list_of_strings)])


def extract_intent_embeddings(model):
    '''
    Extract BERT Embeddings of 'Intent' dataset.
    [model] : BERT model
    '''
    # load 'Intent' dataset
    intent = read_intent_dataset()

    # extract BERT embeddings
    output_file_path = abspath(join(cf.INTENT_DIR, cf.intent_embeddings_bert))
    if is_available(output_file_path):
        print('\tFile already available...')
        return None

    embeddings = encode_bert(list(intent['text']), model)
    print(f'Embeddings shape = {embeddings.shape}')
    np.save(output_file_path, embeddings)

    print('Embeddings extracted and saved.')

    return None


def extract_questions_embeddings(model):
    '''
    Extract BERT Embeddings of 'Questions' dataset.
    [model] : BERT model
    '''
    # load 'Questions' dataset
    questions = read_questions_dataset()

    # extract LaBSE embeddings
    output_file_path = abspath(join(cf.QUESTIONS_DIR,
                                    cf.questions_embeddings_bert))
    if is_available(output_file_path):
        print('\tFile already available...')
        return None

    embeddings = encode_bert(list(questions['text']), model)
    print(f'Embeddings shape = {embeddings.shape}')
    np.save(output_file_path, embeddings)

    print('Embeddings extracted and saved.')

    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--mode', required=True,
                        choices=['intent', 'questions'],
                        help='mode can be either "tweets",'
                             '"intent", or "questions"')
    args = parser.parse_args()

    # read BERT model
    print('Retrieving model...')
    bert_model = Transformer2Vec(cf.bert_model)
    print('Model retrieved.')

    if args.mode == 'intent':
        extract_intent_embeddings(bert_model)
    elif args.mode == 'questions':
        extract_questions_embeddings(bert_model)
    else:
        raise ValueError('Invalid "mode" argument!')
