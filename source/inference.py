'''
Script to run inference given a trained model
'''

__author__ = 'Oguzhan Gencoglu'

from os.path import join
from os.path import abspath
import argparse

from joblib import load
import numpy as np
from glob2 import glob
from tqdm import tqdm

from utils import is_available
from configs import config as cf


def infer_single_batch(batch_number):
    '''
    Run inference on a single batch. Returns probabilities.
    [batch_number] : int
    '''
    print('Processing batch {}'.format(batch_number))
    batch_path = abspath(join(cf.TWEETS_DIR,
                              'compressed_batch{}.npz'.format(batch_number)))
    batch = np.load(batch_path)['arr_0']
    reshaped = np.vstack(np.swapaxes(np.swapaxes(batch, 1, 2), 0, 1))
    probs = model.predict_proba(reshaped)

    return probs


def run_inference(model):
    '''
    Run inference on tweet embeddings and save probabilites
    [model] : scikit-learn model object
    '''

    n_batches = len(glob(abspath(join(cf.TWEETS_DIR, '*.npz'))))
    print('{} batches found...'.format(n_batches))

    for i in tqdm(range(n_batches)):  # iterate through batches
        probs = infer_single_batch(i)
        output_file_path = abspath(join(cf.TWEETS_DIR,
                                        'probabilities{}.npy'.format(i)))
        if is_available(output_file_path):
            print('\tProbabilities already calculated...')
            continue
        np.save(output_file_path, probs)

    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--classifier', required=True,
                        choices=['kNN', 'LR', 'SVM'],
                        help='"kNN", "LR", or "SVM"')
    args = parser.parse_args()

    # load trained model
    model = load(abspath(join(cf.MODELS_DIR,
                              '{}.joblib'.format(args.classifier))))
    print('Model loaded...')
    print(model)

    run_inference(model)
