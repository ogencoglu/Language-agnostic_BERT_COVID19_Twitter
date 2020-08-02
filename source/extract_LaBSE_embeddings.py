'''
Script to extract Language-agnostic Sentence BERT Embeddings (LaBSE)
'''

__author__ = 'Oguzhan Gencoglu'

from os.path import join
from os.path import abspath
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from LaBSE import get_model, get_tokenizer, encode
from utils import (chunks, is_available, read_intent_dataset,
                   read_questions_dataset)
from configs import config as cf


def extract_tweet_embeddings(model, tokenizer):
    '''
    Extract Language-agnostic BERT Sentence Embeddings of tweets.
    Processes in batches to fit into GPU memory.
    [model]     : tf.keras.Model
    [tokenizer] : BERT tokenizer
    '''

    # read tweets
    tweets = pd.read_csv(abspath(join(cf.TWEETS_DIR, cf.cleaned_tweets_file)),
                         header=None, names=['date', 'tweet'])
    tweets.dropna(inplace=True)
    print(tweets.shape)

    # split the tweets list into chunks to be able to fit into GPU memory
    tweet_chunks = list(chunks(list(tweets['tweet']), cf.batch_size))
    print('Data has been split into {} batches of size {}.'.
          format(len(tweet_chunks), cf.batch_size))

    # iterate over batches and extract sentence embeddings
    for batch in tqdm(enumerate(tweet_chunks)):
        print('Processing batch {}...'.format(batch[0]))
        output_file_path = abspath(join(cf.TWEETS_DIR,
                                        'compressed_batch{}.npz'.format(
                                                                    batch[0])))
        if is_available(output_file_path):
            print('\tBatch already extracted...')
            continue

        smaller_chunks = chunks(batch[1], int(cf.batch_size / cf.split_size))
        embeddings = np.dstack(
            [encode(smaller_batch,
                    model,
                    tokenizer).numpy() for smaller_batch in smaller_chunks]
                               )  # 3D numpy array

        np.savez_compressed(output_file_path, embeddings)
    print('Embeddings extracted and saved.')

    return None


def extract_intent_embeddings(model, tokenizer):
    '''
    Extract Language-agnostic BERT Sentence Embeddings of 'Intent' dataset.
    [model]     : tf.keras.Model
    [tokenizer] : BERT tokenizer
    '''
    # load 'Intent' dataset
    intent = read_intent_dataset()

    # extract LaBSE embeddings
    output_file_path = abspath(join(cf.INTENT_DIR, cf.intent_embeddings))
    if is_available(output_file_path):
        print('\tFile already available...')
        return None

    embeddings = encode(list(intent['text']), model, tokenizer).numpy()
    np.save(output_file_path, embeddings)

    print('Embeddings extracted and saved.')

    return None


def extract_questions_embeddings(model, tokenizer):
    '''
    Extract Language-agnostic BERT Sentence Embeddings of 'Questions' dataset.
    [model]     : tf.keras.Model
    [tokenizer] : BERT tokenizer
    '''
    # load 'Questions' dataset
    questions = read_questions_dataset()

    # extract LaBSE embeddings
    output_file_path = abspath(join(cf.QUESTIONS_DIR, cf.questions_embeddings))
    if is_available(output_file_path):
        print('\tFile already available...')
        return None

    embeddings = encode(list(questions['text']), model, tokenizer).numpy()
    np.save(output_file_path, embeddings)

    print('Embeddings extracted and saved.')

    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--mode', required=True,
                        choices=['tweets', 'intent', 'questions'],
                        help='mode can be either "tweets",'
                             '"intent", or "questions"')
    args = parser.parse_args()

    # read LaBSE model
    print('Retrieving model via url...')
    labse_model, labse_layer = get_model(model_url=cf.model_url,
                                         max_seq_length=cf.max_seq_length)
    tokenizer = get_tokenizer(labse_layer)
    print('Model retrieved.')

    if args.mode == 'tweets':
        extract_tweet_embeddings(labse_model, tokenizer)
    elif args.mode == 'intent':
        extract_intent_embeddings(labse_model, tokenizer)
    elif args.mode == 'questions':
        extract_questions_embeddings(labse_model, tokenizer)
    else:
        raise ValueError('Invalid "mode" argument!')
