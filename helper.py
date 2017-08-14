import os
import pickle
import re
from collections import Counter
import numpy as np

def load_data(path, encoding='utf-8'):

    input_file = os.path.join(path)
    with open(input_file, "r", encoding=encoding) as f:
        data = f.read()
    return data

def clean_str(string):

    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def create_lookup_tables(text):

    vocab = set(text)
    vocab_to_int = {c : i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    return vocab_to_int, int_to_vocab

def preprocess_and_save_data(dataset_path, pickle_path, create_lookup_tables, encoding='utf-8'):

    text = load_data(dataset_path, encoding)
    #text = clean_str(text)
    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
    print(len(int_text))
    with open(pickle_path, 'wb') as f:
        pickle.dump((int_text, vocab_to_int, int_to_vocab), f)


def load_preprocess(pickle_path):

    with open(pickle_path, mode='rb') as f:
        return pickle.load(f)

def explore_data(text):

    view_sentence_range = (0, 50)
    print('Dataset Stats\n')
    print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
    scenes = text.split('\n\n\n')
    print('Number of scenes: {}'.format(len(scenes)))
    sentence_count_scene = [scene.count('\n') for scene in scenes]
    print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

    sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
    print('Number of lines: {}'.format(len(sentences)))
    word_count_sentence = [len(sentence.split()) for sentence in sentences]
    print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))
    print()
    print('The sentences {} to {}:'.format(*view_sentence_range))
    print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))

def next_batch(arr, batch_size, seq_length):

    no_batches = len(arr) // (batch_size * seq_length)
    arr = arr[:no_batches * batch_size * seq_length]
    arr = arr.reshape((batch_size,-1))

    for i in range(0, arr.shape[1], seq_length):
        x = arr[:, i:i+seq_length]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y


if __name__ == '__main__':

    encoding = "ISO-8859-1"#'utf-8'
    script = input('Enter Name of script to preprocess: ')
    path = './data/' + script + '.txt'
    text = load_data(path, encoding)
    explore_data(text)
    pickle_path = './model/' + script + '/' + script + '_preprocess.p'
    preprocess_and_save_data(path, pickle_path, create_lookup_tables, encoding)