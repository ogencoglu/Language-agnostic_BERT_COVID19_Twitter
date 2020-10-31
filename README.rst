.. image:: https://github.com/ogencoglu/Language-agnostic_BERT_COVID19_Twitter/blob/master/media/timeline.png
   :width: 400

Implementation of `Large-scale, Language-agnostic Discourse Classification of Tweets During COVID-19 <https://arxiv.org/abs/2008.00461>`_ - Gencoglu O. (2020)
====================
This repository provides the full implementation in *python 3.7*. Requires Twitter developer account.

Main Idea
====================
**Utilizing Language-agnostic BERT Sentence Embeddings (LaBSE) to analyze 28 million tweets in 109 languages related to COVID-19**

.. raw:: html

    <img src="https://github.com/ogencoglu/Language-agnostic_BERT_COVID19_Twitter/blob/master/media/umap.png" height="700px" class="center">

Reproduction of Results
====================

Follow steps *1-5* below.

1 - Get the Data
--------------

See *directory_info* in the *data* directory for the expected files.

**1.1 -** `Download <https://zenodo.org/record/3738018#.Xya8tGMzbCJ>`_ 30+ million tweet IDs and hydrate them into timestamp and tweet text (requires Twitter developer account). 

.. code-block:: bash

   Jan 17,tweet_text_string
   Jan 27,tweet_text_string
   ...

Once *tweets.csv* is in the example format above, preprocess by running:

.. code-block:: bash

  python3.7 preprocess.py

**1.2 -** Download *Intent* and *Questions* datasets

  --Intent Dataset        `Link <https://fb.me/covid_mcid_dataset>`_
  --Questions Dataset     `Link <https://github.com/JerryWei03/COVID-Q>`_

2 - Extract Tweet Embeddings
-------------------------------

2.1 - **BERT**

.. code-block:: bash

  python3.7 extract_BERT_embeddings.py -m intent
  python3.7 extract_BERT_embeddings.py -m questions

2.2 - **Language-agnostic BERT Sentence Embeddings (LaBSE)**

.. code-block:: bash

  python3.7 extract_LaBSE_embeddings.py -m tweets
  python3.7 extract_LaBSE_embeddings.py -m intent
  python3.7 extract_LaBSE_embeddings.py -m questions

Relevant configurations are defined in *configs.py*, e.g.:

  --model_url                  'https://tfhub.dev/google/LaBSE/1'
  --max_seq_length             128
  --bert_model                 'bert-base-multilingual-uncased'

3 - Cross-validation and Bayesian Hyperparameter Optimization
-------------------------------

.. code-block:: bash

  python3.7 train.py -m hyper_opt -c "model_identifier" -e "embeddings_identifier"

4 - Train
-------------------------------

.. code-block:: bash

  python3.7 train.py -m train -c "model_identifier"

5 - Inference
-------------------------------

.. code-block:: bash

  python3.7 inference.py -c "model_identifier"

*source* directory tree:

.. code-block:: bash

  ├── configs.py
  ├── extract_BERT_embeddings.py
  ├── extract_LaBSE_embeddings.py
  ├── inference.py
  ├── LaBSE.py
  ├── preprocess.py
  ├── train.py
  ├── umap_vis.py
  └── utils.py
  
`Cite <https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Large-scale%2C+Language-agnostic+Discourse+Classification+of+Tweets+During+COVID-19&btnG=>`_
====================
  
.. code-block::

    @article{gencoglu2020large,
      title={Large-scale, Language-agnostic Discourse Classification of Tweets During COVID-19},
      author={Gencoglu, Oguzhan},
      journal={arXiv preprint arXiv:2008.00461},
      year={2020}
    }
    
Or

    Gencoglu, Oguzhan. "Large-scale, Language-agnostic Discourse Classification of Tweets During COVID-19." arXiv preprint arXiv:2008.00461 (2020).
