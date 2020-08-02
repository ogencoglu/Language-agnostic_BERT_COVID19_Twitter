'''
UMAP dimensionality reduction for visualization
'''

__author__ = 'Oguzhan Gencoglu'

import umap

from configs import config as cf


def embed_umap(data):
    '''
    Returns 2D UMAP embeddings of high dimensional data
    [data] : 2D numpy array, shape = (n_samples, n_features)
    '''
    umap_embedding = umap.UMAP(n_neighbors=5,
                               min_dist=0.25,
                               metric=cf.distance_metric).fit_transform(data)

    return umap_embedding
