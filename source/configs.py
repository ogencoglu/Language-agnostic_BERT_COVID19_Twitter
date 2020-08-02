'''
configs & settings are defined in this file
'''

__author__ = 'Oguzhan Gencoglu'

from os.path import join
from os.path import abspath
from os.path import dirname
from os import pardir


class Config(object):

    # _________________ Common to all experiments _________________

    # directory paths
    CURRENT_DIR = abspath(dirname(__file__))
    ROOT_DIR = abspath(join(CURRENT_DIR, pardir))
    DATA_DIR = abspath(join(ROOT_DIR, 'data'))
    TWEETS_DIR = abspath(join(DATA_DIR, 'tweets'))
    INTENT_DIR = abspath(join(DATA_DIR, 'intent'))
    QUESTIONS_DIR = abspath(join(DATA_DIR, 'questions'))
    MODELS_DIR = abspath(join(ROOT_DIR, 'models'))
    LOGS_DIR = abspath(join(ROOT_DIR, 'logs'))

    # file paths
    tweets_file = 'tweets.csv'
    cleaned_tweets_file = 'tweets_cleaned.csv'
    questions_embeddings = 'questions_embeddings.npy'
    intent_embeddings = 'intent_embeddings.npy'

    # language agnostic sentence BERT params
    model_url = "https://tfhub.dev/google/LaBSE/1"
    max_seq_length = 128

    # params related to embedding extraction processs
    batch_size = 50000
    split_size = 50

    # 'intent' dataset related params
    intent_label_map = {
        'intent:what_are_symptoms': 'Symptoms',
        'intent:what_are_treatment_options': 'Treatment',
        'intent:how_does_corona_spread': 'Transmission',
        'intent:can_i_get_from_feces_animal_pets': 'Transmission',
        'intent:can_i_get_from_packages_surfaces': 'Transmission',
        'intent:protect_yourself': 'Prevention',
        'intent:latest_numbers': 'Reporting',
        'intent:myths': 'Speculation',
        'intent:what_if_i_visited_high_risk_area': 'Travel',
        'intent:travel': 'Travel',
        'intent:news_and_press': 'News & Press',
        'intent:what_is_corona': 'What Is Corona?',
        'intent:share': 'Share',
        'intent:donate': 'Donate'
    }

    # 'question' dataset related params
    questions_relevant_categories = ['Symptoms', 'Treatment', 'Transmission',
                                     'Prevention', 'Reporting', 'Speculation']

    # Bayesian hyper-parameter optimization
    n_iters = 30
    n_cv_folds = 10  # number of cross-validation folds
    n_jobs = -1  # use all available cores

    # UMAP
    distance_metric = 'cosine'


config = Config()
