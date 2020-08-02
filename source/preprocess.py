'''
Script to clean tweets
'''

__author__ = 'Oguzhan Gencoglu'

from os.path import join
from os.path import abspath

from tqdm import tqdm
import pandas as pd

from configs import config as cf


def clean_tweet(text):
    '''
    clean tweets by removing usernames and urls
    [text] : str
    '''

    without_usernames = []
    for s in text.split():
        if '@' not in s:
            without_usernames.append(s)
    text = ' '.join(without_usernames)

    without_url = []
    for s in text.split():
        if 'http' not in s:
            without_url.append(s)
    text = ' '.join(without_url)

    text = text.strip()
    return text


if __name__ == '__main__':

    df = pd.read_csv(abspath(join(cf.TWEETS_DIR, cf.tweets_file)),
                     header=None,
                     names=['date', 'tweet'])

    tqdm.pandas()
    df['tweet'] = df['tweet'].progress_apply(clean_tweet)

    df.to_csv(abspath(join(cf.TWEETS_DIR, cf.cleaned_tweets_file)),
              index=False,
              header=False)
