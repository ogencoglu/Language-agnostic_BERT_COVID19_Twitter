'''
Language-agnostic Sentence BERT Embeddings (LaBSE) utilities
'''

__author__ = 'Oguzhan Gencoglu'

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import bert

from configs import config as cf


def get_model(model_url, max_seq_length):
    '''
    loads model given a valid url and maximum sequence length
    [model_url]      : tensorflow hub model URL
    [max_seq_length] : int
    '''
    labse_layer = hub.KerasLayer(model_url, trainable=True)

    # Define input
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,),
                                           dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,),
                                       dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,),
                                        dtype=tf.int32,
                                        name="segment_ids")

    # LaBSE layer
    pooled_output,  _ = labse_layer([input_word_ids, input_mask, segment_ids])

    # The embedding is l2 normalized
    pooled_output = tf.keras.layers.Lambda(
                            lambda x: tf.nn.l2_normalize(x))(pooled_output)

    return tf.keras.Model(
        inputs=[input_word_ids, input_mask, segment_ids],
        outputs=pooled_output), labse_layer


def create_input(input_list, tokenizer, max_seq_length):
    '''
    BERT-style input preparation
    [input_list]     : list of strings
    [tokenizer]      : BERT tokenizer
    [max_seq_length] : int
    '''
    input_ids_all, input_mask_all, segment_ids_all = [], [], []
    for input_string in input_list:
        # Tokenize input
        input_tokens = ["[CLS]"] + tokenizer.tokenize(input_string) + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        sequence_length = min(len(input_ids), max_seq_length)

        # Pad or clip
        if len(input_ids) >= max_seq_length:
            input_ids = input_ids[:max_seq_length]
        else:
            input_ids = input_ids + [0] * (max_seq_length - len(input_ids))

        input_mask = [1] * sequence_length + [0] * (
                                            max_seq_length - sequence_length)

        input_ids_all.append(input_ids)
        input_mask_all.append(input_mask)
        segment_ids_all.append([0] * max_seq_length)

    input_ids_all = np.array(input_ids_all)
    input_mask_all = np.array(input_mask_all)
    segment_ids_all = np.array(segment_ids_all)

    return input_ids_all, input_mask_all, segment_ids_all


def get_tokenizer(embed_layer):
    '''
    return tokenizer given the LaBSE layer
    [embed_layer] : tensorflow_hub.keras_layer
    '''
    vocab_file = embed_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = embed_layer.resolved_object.do_lower_case.numpy()
    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

    return tokenizer


def encode(input_text, model, tokenizer):
    '''
    returns embeddings of size [batch_size, 768]
    [input_text] : list of strings
    [model]      : tf.keras.Model
    [tokenizer]  : BERT tokenizer
    '''
    input_ids, input_mask, segment_ids = create_input(
        input_text, tokenizer, cf.max_seq_length)

    return model([input_ids, input_mask, segment_ids])
