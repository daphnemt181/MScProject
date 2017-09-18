# -*- coding: utf-8 -*-

"""
Various data processing functions to be used in other classes
"""


import sys
import os
import pickle
import random
import time
from collections import Counter
from nltk.tokenize import word_tokenize
import string
import re
import codecs

sys.path.append('../../../')

import SETTINGS


def chunker(arrays, chunk_size):
    """Split the arrays into equally sized chunks

    :param arrays: (N * L np.array) arrays of sentences
    :param chunk_size: (int) size of the chunks

    :return chunks: (list) list of the chunks, numpy arrays"""
    chunks = []

    # sentence length
    sent_len = len(arrays[0])

    # chunk the arrays over the sentence in chunk sizes
    for pos in range(0, sent_len, chunk_size):
        chunk = []

        for array in arrays:
            chunk.append(array[pos: pos+chunk_size])

        chunks.append(chunk)

    return chunks

def char2int(sentence, valid_vocab):
    """Takes a sentence from the data set and turns it into index form

    Sentence is a list that has characters and turns this into an index form
    by using the valid_vocab list, which is a list of the vocabulary characters,
    as a lookup table. Note that due to how we handle <EOS> and filler characters,
    filler character must map back to integer 0.

    :param sentence: list of characters of length l
    :param valid_vocab: list of characters of length D

    :return: list of integers of length l"""
    sentence_int = []

    for char in sentence:
        index = valid_vocab.index(char)
        sentence_int.append(index)

    return sentence_int

def int2char(sentence, valid_vocab):
    """Takes a sentence in the integer form and turns it into character form

    Sentence is a list that has integers representing index of the character in the
    valid_vocab list. Use the valid_vocab list to get the sentence in character form.

    :param sentence: list of integers of length l
    :param valid_vocab: list of characters of length D, by convention
    first character is the filler 'F' and last character is end-of-sentence 'E'

    :return: list of character of length l"""
    sentence_char = []

    for i in sentence:
        if valid_vocab[i] == 'E':
            break
        sentence_char.append(valid_vocab[i])

    return sentence_char

def preprocess_europarl_enfr(save_name='full', process_size=None):
    """Preprocess the dataset

    Convert everything to lower case. Remove all of the sentences which includes invalid characters,
    characters not in the valid_vocab list. Transform it into integer form. Get it into a form of
    [[2, 4], [2, 1, 5]] etc.

    :param save_name: (str) save name of the set
    :param process_size: (int) the number of sentences to process"""
    # paths directory
    PATHS = SETTINGS.PATHS

    # valid vocabulary
    valid_vocab_fr = SETTINGS.valid_vocab_fr
    valid_vocab_en = SETTINGS.valid_vocab_en

    # open files
    fr_file = codecs.open(os.path.join(PATHS['data_raw_dir'], 'europarl-v7.fr-en.fr'), 'r', 'utf-8')
    en_file = codecs.open(os.path.join(PATHS['data_raw_dir'], 'europarl-v7.fr-en.en'), 'r', 'utf-8')

    # get all sentences as strings
    fr_data_set = fr_file.readlines()
    en_data_set = en_file.readlines()

    # close files
    fr_file.close()
    en_file.close()

    # size of dataset. If size is none use whole data set or if bigger than the size else use passed value
    process_size = len(fr_data_set) if process_size is None or process_size >= len(fr_data_set) else process_size

    # make all lines lowercase
    # strip of newlines
    # make into list
    # append end character
    fr_data_set = [list(line.lower().strip()) + [u'E'] for line in fr_data_set[0:process_size]]
    en_data_set = [list(line.lower().strip()) + [u'E'] for line in en_data_set[0:process_size]]

    # loop over both sentences and make sure that both are valid
    # i.e. all characters in the valid vocabulary set for both source and
    # sentence. Convert it to vocab form.
    fr_data_set_post = []
    en_data_set_post = []

    for fr, en in zip(fr_data_set, en_data_set):
        # If all of the characters in the french and the english line are in
        # respective valid vocabulary, add them to the post processed data set
        if set(fr).issubset(set(valid_vocab_fr)) and set(en).issubset(set(valid_vocab_en)):
            fr_data_set_post.append(char2int(fr, valid_vocab_fr))
            en_data_set_post.append(char2int(en, valid_vocab_en))

    print('Creating save directory.')
    out_dir = os.path.join(PATHS['data_processed_dir'], 'fr_en_chars_{}'.format(save_name))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # dump the processed data to file
    pickle.dump(fr_data_set_post, open(os.path.join(out_dir, 'fr_processed.pickle'), 'wb'))
    pickle.dump(en_data_set_post, open(os.path.join(out_dir, 'en_processed.pickle'), 'wb'))


def preprocess_europarl_enfr_words(save_name='full', process_size=None, vocab_size=50000):
    """Preprocess the dataset

    Convert everything to lower case. Remove all of the sentences which includes invalid characters,
    characters not in the valid_vocab list. Transform it into integer form. Get it into a form of
    [[2, 4], [2, 1, 5]] etc.

    :param save_name: (str) save name of the set
    :param process_size: (int) the number of sentences to process"""
    # paths directory
    PATHS = SETTINGS.PATHS

    # min and max sentence length
    min_len = 2
    max_len = 30

    # open files
    fr_file = codecs.open(os.path.join(PATHS['data_raw_dir'], 'europarl-v7.fr-en.fr'), 'r', 'utf-8')
    en_file = codecs.open(os.path.join(PATHS['data_raw_dir'], 'europarl-v7.fr-en.en'), 'r', 'utf-8')

    # get all sentences as strings
    fr_data_set = fr_file.readlines()
    en_data_set = en_file.readlines()

    # close files
    fr_file.close()
    en_file.close()

    # size of dataset. If size is none use whole data set or if bigger than the size else use passed value
    process_size = len(fr_data_set) if process_size is None or process_size >= len(fr_data_set) else process_size

    fr_data_set = fr_data_set[0:process_size]
    en_data_set = en_data_set[0:process_size]

    # Compute word frequencies and select french and english vocabulary
    print('Compute word frequencies')

    start = time.clock()
    num_sentences_processed = 0

    word_counts_fr = Counter()
    word_counts_en = Counter()

    word_counts_fr.update(word_tokenize(' '.join(fr_data_set).lower()))
    word_counts_en.update(word_tokenize(' '.join(en_data_set).lower()))

    print('word frequencies computed; time taken = ' + str(time.clock() - start) + ' seconds')

    # create vocabularies based on frequency of words
    valid_vocab_fr = list(dict(word_counts_fr.most_common(vocab_size)).keys())
    valid_vocab_counts_fr = list(dict(word_counts_fr.most_common(vocab_size)).values())

    valid_vocab_en = list(dict(word_counts_en.most_common(vocab_size)).keys())
    valid_vocab_counts_en = list(dict(word_counts_en.most_common(vocab_size)).values())

    print('valid vocab examples')
    for i in range(5):
        print(u'{:<10}  {:<10}'.format(unicode(valid_vocab_en[i]), unicode(valid_vocab_fr[i])))

    unk_token = '<UNK>'
    valid_vocab_fr.append(unk_token)
    valid_vocab_en.append(unk_token)
    valid_vocab_counts_fr.append(0)
    valid_vocab_counts_en.append(0)

    eos_token = '<EOS>'
    valid_vocab_fr = [eos_token] + valid_vocab_fr
    valid_vocab_en = [eos_token] + valid_vocab_en
    valid_vocab_counts_fr = [0] + valid_vocab_counts_fr
    valid_vocab_counts_en = [0] + valid_vocab_counts_en

    valid_vocab_index_fr = {valid_vocab_fr[i]: i for i in range(len(valid_vocab_fr))}
    valid_vocab_index_en = {valid_vocab_en[i]: i for i in range(len(valid_vocab_en))}

    print('Creating save directory.')
    out_dir = os.path.join(PATHS['data_processed_dir'], 'fr_en_{}to{}_vocab{}_{}'.format(min_len, max_len, vocab_size, save_name))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # save valid vocabularies
    pickle.dump(valid_vocab_fr, open(os.path.join(out_dir, 'fr_valid_vocab.pickle'), 'wb'))
    pickle.dump(valid_vocab_en, open(os.path.join(out_dir, 'en_valid_vocab.pickle'), 'wb'))

    # make all lines lowercase
    # strip of newlines
    # tokenise
    # replace words not in the vocabulary with the <UNK> token
    # append end-of-sentence token
    fr_sentences = []
    en_sentences = []

    for fr, en in zip(fr_data_set, en_data_set):
        fr = fr.strip().lower()
        en = en.strip().lower()
        fr_tokens = word_tokenize(fr)
        en_tokens = word_tokenize(en)

        if len(fr_tokens) < min_len or len(fr_tokens) > max_len or len(en_tokens) < min_len or len(en_tokens) > max_len:
            continue

        fr_indexed = []
        en_indexed = []

        for w in en_tokens:
            try:
                en_indexed.append(valid_vocab_index_en[w])
            except KeyError:
                en_indexed.append(valid_vocab_index_en[unk_token])
                valid_vocab_counts_en[-1] += 1

        en_indexed.append(valid_vocab_index_en[eos_token])
        valid_vocab_counts_en[0] += 1

        for w in fr_tokens:
            try:
                fr_indexed.append(valid_vocab_index_fr[w])
            except KeyError:
                fr_indexed.append(valid_vocab_index_fr[unk_token])
                valid_vocab_counts_fr[-1] += 1

        fr_indexed.append(valid_vocab_index_fr[eos_token])
        valid_vocab_counts_fr[0] += 1

        en_sentences.append(en_indexed)
        fr_sentences.append(fr_indexed)

        num_sentences_processed += 1

        if num_sentences_processed % 100000 == 0:
            print(str(num_sentences_processed) + ' sentences processed; time taken = ' + str(time.clock() - start) +
                  ' seconds')

    print('There are ' + str(len(fr_sentences)) + ' sentences.')

    # dump the valid vocab counts to file
    pickle.dump(valid_vocab_counts_fr, open(os.path.join(out_dir, 'fr_valid_vocab_counts.pickle'), 'wb'))
    pickle.dump(valid_vocab_counts_en, open(os.path.join(out_dir, 'en_valid_vocab_counts.pickle'), 'wb'))

    # dump the processed data to file
    pickle.dump(fr_sentences, open(os.path.join(out_dir, 'fr_word_level_processed.pickle'), 'wb'))
    pickle.dump(en_sentences, open(os.path.join(out_dir, 'en_word_level_processed.pickle'), 'wb'))

def preprocess_all_words(save_name='europarl_un_nc', vocab_size=50000):
    """Preprocess the dataset

    Convert everything to lower case. Remove all of the sentences which includes invalid characters,
    characters not in the valid_vocab list. Transform it into integer form. Get it into a form of
    [[2, 4], [2, 1, 5]] etc.

    :param save_name: (str) save name of the set
    :param process_size: (int) the number of sentences to process"""
    # paths directory
    PATHS = SETTINGS.PATHS

    # min and max sentence length
    min_len = 2
    max_len = 30

    en_dataset_path = os.path.join(PATHS['data_raw_dir'], 'full_dataset', 'en')
    fr_dataset_path = os.path.join(PATHS['data_raw_dir'], 'full_dataset', 'fr')

    en_filenames = sorted(os.listdir(en_dataset_path))
    fr_filenames = sorted(os.listdir(fr_dataset_path))

    word_counts_en = Counter()
    word_counts_fr = Counter()

    num_sentences_processed = 0

    for en_filename, fr_filename in zip(en_filenames, fr_filenames):
        print('On files: {}, {}'.format(en_filename, fr_filename))
        # open files
        en_file = codecs.open(os.path.join(en_dataset_path, en_filename), 'r', 'utf-8')
        fr_file = codecs.open(os.path.join(fr_dataset_path, fr_filename), 'r', 'utf-8')

        # get all sentences as strings
        en_data = en_file.readlines()
        fr_data = fr_file.readlines()

        # close files
        en_file.close()
        fr_file.close()

        # Compute word frequencies and select french and english vocabulary
        print('Compute word frequencies')

        start = time.clock()

        word_counts_en.update(word_tokenize(' '.join(en_data).lower()))
        word_counts_fr.update(word_tokenize(' '.join(fr_data).lower()))

        print('word frequencies computed; time taken = ' + str(time.clock() - start) + ' seconds')

    # create vocabularies based on frequency of words
    valid_vocab_fr = list(dict(word_counts_fr.most_common(vocab_size)).keys())
    valid_vocab_counts_fr = list(dict(word_counts_fr.most_common(vocab_size)).values())

    valid_vocab_en = list(dict(word_counts_en.most_common(vocab_size)).keys())
    valid_vocab_counts_en = list(dict(word_counts_en.most_common(vocab_size)).values())

    unk_token = '<UNK>'
    valid_vocab_fr.append(unk_token)
    valid_vocab_en.append(unk_token)
    valid_vocab_counts_fr.append(0)
    valid_vocab_counts_en.append(0)

    eos_token = '<EOS>'
    valid_vocab_fr = [eos_token] + valid_vocab_fr
    valid_vocab_en = [eos_token] + valid_vocab_en
    valid_vocab_counts_fr = [0] + valid_vocab_counts_fr
    valid_vocab_counts_en = [0] + valid_vocab_counts_en

    valid_vocab_index_fr = {valid_vocab_fr[i]: i for i in range(len(valid_vocab_fr))}
    valid_vocab_index_en = {valid_vocab_en[i]: i for i in range(len(valid_vocab_en))}

    print('Creating save directory.')
    out_dir = os.path.join(PATHS['data_processed_dir'], 'fr_en_{}to{}_vocab{}_{}'.format(min_len, max_len, vocab_size, save_name))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # save valid vocabularies
    pickle.dump(valid_vocab_fr, open(os.path.join(out_dir, 'fr_valid_vocab.pickle'), 'wb'))
    pickle.dump(valid_vocab_en, open(os.path.join(out_dir, 'en_valid_vocab.pickle'), 'wb'))

    # make all lines lowercase
    # strip of newlines
    # tokenise
    # replace words not in the vocabulary with the <UNK> token
    # append end-of-sentence token
    fr_sentences = []
    en_sentences = []

    print('Creating data (sentences)')

    for en_filename, fr_filename in zip(en_filenames, fr_filenames):
        print('On files: {}, {}'.format(en_filename, fr_filename))
        # open files
        en_file = codecs.open(os.path.join(en_dataset_path, en_filename), 'r', 'utf-8')
        fr_file = codecs.open(os.path.join(fr_dataset_path, fr_filename), 'r', 'utf-8')

        # get all sentences as strings
        fr_data = fr_file.readlines()
        en_data = en_file.readlines()

        # close files
        fr_file.close()
        en_file.close()

        for fr, en in zip(fr_data, en_data):
            fr = fr.strip().lower()
            en = en.strip().lower()
            fr_tokens = word_tokenize(fr)
            en_tokens = word_tokenize(en)

            if len(fr_tokens) < min_len or len(fr_tokens) > max_len or len(en_tokens) < min_len or len(en_tokens) > max_len:
                continue

            fr_indexed = []
            en_indexed = []

            for w in en_tokens:
                try:
                    en_indexed.append(valid_vocab_index_en[w])
                except KeyError:
                    en_indexed.append(valid_vocab_index_en[unk_token])
                    valid_vocab_counts_en[-1] += 1

            en_indexed.append(valid_vocab_index_en[eos_token])
            valid_vocab_counts_en[0] += 1

            for w in fr_tokens:
                try:
                    fr_indexed.append(valid_vocab_index_fr[w])
                except KeyError:
                    fr_indexed.append(valid_vocab_index_fr[unk_token])
                    valid_vocab_counts_fr[-1] += 1

            fr_indexed.append(valid_vocab_index_fr[eos_token])
            valid_vocab_counts_fr[0] += 1

            en_sentences.append(en_indexed)
            fr_sentences.append(fr_indexed)

            num_sentences_processed += 1

            if num_sentences_processed % 100000 == 0:
                print(str(num_sentences_processed) + ' sentences processed; time taken = ' + str(time.clock() - start) + ' seconds')

    print('There are ' + str(len(fr_sentences)) + ' sentences.')

    # dump the valid vocab counts to file
    pickle.dump(valid_vocab_counts_fr, open(os.path.join(out_dir, 'fr_valid_vocab_counts.pickle'), 'wb'))
    pickle.dump(valid_vocab_counts_en, open(os.path.join(out_dir, 'en_valid_vocab_counts.pickle'), 'wb'))

    # same seed for both sentences
    seed = 1993

    # Use Shafer-Yates to shuffle in place
    random.seed(seed)
    random.shuffle(en_sentences)
    random.seed(seed)
    random.shuffle(fr_sentences)

    # dump the processed data to file
    pickle.dump(fr_sentences, open(os.path.join(out_dir, 'fr_word_level_processed.pickle'), 'wb'))
    pickle.dump(en_sentences, open(os.path.join(out_dir, 'en_word_level_processed.pickle'), 'wb'))


if __name__ == '__main__':
    # preprocess_europarl_enfr(save_name='20000', process_size=10000)
    preprocess_all_words(save_name='un_europarl_nc', vocab_size=30000)
