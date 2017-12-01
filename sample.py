from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import helper
from model import *

script = 'code'

pickle_path = './model/' + script + '/' + script + '_preprocess.p'
save_dir = './model/' + script + '/'

_, vocab_to_int, int_to_vocab = helper.load_preprocess(pickle_path)
vocab_size = len(vocab_to_int.keys())

#Hyperparameters.
num_epochs  = 100
batch_size  = 20
rnn_size    = 256
num_layers  = 2
embed_dim   = 256
seq_length  = 200
lr          = 0.001
keep_prob   = 0.5
temperature = 1.0
grad_clip   = 5
decay_rate  = 0.97

def pick_char(preds, N=5):
    global vocab_size
    tmp = np.squeeze(preds)
    tmp[np.argsort(tmp)[:-N]] = 0
    tmp = tmp / np.sum(tmp)
    c = np.random.choice(vocab_size, 1, p=tmp)[0]
    return c

def sample(checkpoint, gen_length, prime="The "):
    samples = [c for c in prime]
    nn = text_rnn(batch_size, seq_length, rnn_size, num_layers, vocab_size, grad_clip, False)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        state = sess.run(nn.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = vocab_to_int[c]
            feed = {
                    nn.inputs: x,
                    nn.keep_prob: keep_prob,
                    nn.learning_rate: lr,
                    nn.initial_state: state,
                    nn.temp: temperature
                    }

            preds, state = sess.run([nn.probs, nn.final_state], feed_dict=feed)
        c = pick_char(preds)
        samples.append(int_to_vocab[c])

        for i in range(gen_length):
            x[0,0] = c
            feed = {
                    nn.inputs: x,
                    nn.keep_prob: keep_prob,
                    nn.learning_rate: lr,
                    nn.initial_state: state,
                    nn.temp: temperature
                    }

            preds, state = sess.run([nn.probs, nn.final_state], feed_dict=feed)

            c = pick_char(preds)
            samples.append(int_to_vocab[c])

    return ''.join(samples)

if __name__ == '__main__':
    checkpoint = tf.train.latest_checkpoint(save_dir)
    text = sample(checkpoint, 2000, prime="def ")
    print(text)