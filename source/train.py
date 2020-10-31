'''
Script to perform hyper-parameter search and train a classifier
'''

__author__ = 'Oguzhan Gencoglu'

from os.path import join
from os.path import abspath
import argparse

from joblib import dump
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args

from utils import merge_datasets, save_logs, load_logs
from configs import config as cf


def get_classifier(classifier_identifier):
    '''
    Initiates a model
    [classifier_identifier] : str, one of 'kNN', 'LR', or 'SVM'
    '''
    if classifier_identifier == 'kNN':
        classifier = Pipeline([('scaler', StandardScaler()),
                               ('kNN',
                                KNeighborsClassifier(algorithm='brute'))])
    elif classifier_identifier == 'LR':
        classifier = LogisticRegression(max_iter=200)
    elif classifier_identifier == 'SVM':
        classifier = SVC()
    else:
        ValueError('Invalid classifier identifier!')

    return classifier


def get_param_space(classifier_identifier):
    '''
    Returns hyperpameter seach space with prior probability distributions
    [classifier_identifier] : str, one of 'kNN', 'LR', or 'SVM'
    '''
    if classifier_identifier == 'kNN':
        param_space = [
            Categorical(np.arange(1, 16, 2), name='kNN__n_neighbors'),
            Categorical(['cosine', 'euclidean', 'manhattan'],
                        name='kNN__metric'),
        ]
    elif classifier_identifier == 'LR':
        param_space = [
            Real(1e2, 1e4, prior="log-uniform", name='C'),
        ]
    elif classifier_identifier == 'SVM':
        param_space = [
            Categorical(['rbf', 'linear', 'poly'], name='kernel'),
            Real(0.1, 10, prior="log-uniform", name='C'),
        ]
    else:
        ValueError('Invalid classifier identifier!')

    return param_space


def gaussian_process_hyper_opt(data, labels, model, param_space,
                               model_name):
    '''
    Bayesian Optimization of classifier hyper-parameters using Gaussian.
    Processes. Saves the best parameters to "logs" directory.
    [data]        : numpy array
    [labels]      : list or numpy array
    [model]       : scikit-learn model object
    [param_space] : list
    [model_name]  : str
    '''

    @use_named_args(param_space)
    def objective(**params):
        '''
        objective function to minimize
        '''
        model.set_params(**params)
        scores = cross_val_score(estimator=model, X=data, y=labels,
                                 cv=cf.n_cv_folds, n_jobs=cf.n_jobs, verbose=0)
        return -1 * np.mean(scores)

    # start Bayesian optimization
    gaussian_process = gp_minimize(objective, dimensions=param_space,
                                   n_calls=cf.n_iters, verbose=True)
    print('\tHyper-parameter search completed...')

    # found best hyper-parameters
    best_params = {
        parameter.name: gaussian_process.x[i] for i, parameter in enumerate(
                                                    gaussian_process['space'])}
    print('Best hyperparameters:', best_params)

    # save best params to LOGS
    save_logs(best_params, model_name)

    return best_params


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--mode', required=True,
                        choices=['hyper_opt', 'train'],
                        help='mode can be either "hyper_opt" or "train"')
    parser.add_argument('-c', '--classifier', required=True,
                        choices=['kNN', 'LR', 'SVM'],
                        help='"kNN", "LR", or "SVM"')
    parser.add_argument('-e', '--embeddings', required=True,
                        choices=['bert', 'labse'],
                        help='"bert" or "labse"')
    args = parser.parse_args()

    # load data
    data, embeddings = merge_datasets(args.embeddings)
    labels = LabelEncoder().fit_transform(data['class'])

    # instantiate classifier model
    model = get_classifier(args.classifier)

    if args.mode == 'hyper_opt':
        param_space = get_param_space(args.classifier)
        gaussian_process_hyper_opt(data=embeddings, labels=labels, model=model,
                                   param_space=param_space,
                                   model_name='{}_{}'.format(args.classifier,
                                                             args.embeddings))
    elif args.mode == 'train':
        print('\nTraining for {}...'.format(args.classifier))
        best_params = load_logs(args.classifier)
        if args.classifier == 'SVM':
            best_params['probability'] = True
        print('\tBest_params:', best_params)

        model.set_params(**best_params)
        model.fit(embeddings, labels)
        print('Training completed...')
        dump(model, abspath(join(cf.MODELS_DIR,
                                 '{}.joblib'.format(args.classifier))))
    else:
        raise ValueError('Invalid "mode" argument!')
