from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import numpy as np
import tensorflow as tf
import logging
from qa_model import QASystem, Config
from os.path import join as pjoin
from tensorflow.python.platform import gfile
from tensorflow.python import debug as tf_debug

from data_helper import DataLoader
from util import *

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.0004, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.9, 
    "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 300, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", 
    "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", 
    "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", 
    "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{vocab_dim}.npz)")
tf.app.flags.DEFINE_string("train_summary_dir", "train_summary",
    "Directory for saving training summary files")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        print("Restored variables from checkpoint")
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def load_main_data():
    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    
    vocab, rev_vocab = initialize_vocab(vocab_path)
    data = np.load(embed_path)
    glove_vectors = data['glove'].astype(np.float32)

    return glove_vectors, vocab, rev_vocab


def load_train_data(question_len=None, context_len=None):
    data_path = pjoin(FLAGS.data_dir + "/train.all.npz")

    if not gfile.Exists(data_path):
        dl = DataLoader(FLAGS.data_dir, 
                        "train.ids.question", "train.ids.context", "train.span", 
                        FLAGS.data_dir, "train.all")
        dl.load_and_save(question_len=question_len, context_len=context_len)

    data = np.load(data_path)
    question_ids = data['question_ids']
    context_ids = data['context_ids']
    answer_ids = data['answer_ids']
    question_masks = data['question_masks']
    context_masks = data['context_masks']
    question_len = question_ids.shape[1]
    context_len = context_ids.shape[1]

    return question_ids, context_ids, answer_ids, question_len, context_len, question_masks, context_masks


def load_val_data(question_len=None, context_len=None):
    data_path = pjoin(FLAGS.data_dir + "/val.all.npz")

    if not gfile.Exists(data_path):
        dl = DataLoader(FLAGS.data_dir, 
                        "val.ids.question", "val.ids.context", "val.span", 
                        FLAGS.data_dir, "val.all")
        dl.load_and_save(question_len=question_len, context_len=context_len)

    data = np.load(data_path)
    question_ids = data['question_ids']
    context_ids = data['context_ids']
    answer_ids = data['answer_ids']
    question_masks = data['question_masks']
    context_masks = data['context_masks']

    return question_ids, context_ids, answer_ids, question_masks, context_masks


def main(_):
    glove_vectors, vocab, rev_vocab = load_main_data()

    train_question_ids, train_context_ids, train_answer_ids, \
    question_len, context_len, train_question_masks, train_context_masks = load_train_data()
    
    val_question_ids, val_context_ids, val_answer_ids, \
    val_question_masks, val_context_masks = load_val_data(question_len=question_len, 
                                                        context_len=context_len)

    global_train_dir = '/tmp/cs224n-squad-train'
    # Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    # file paths saved in the checkpoint. This allows the model to be reloaded even
    # if the location of the checkpoint files has moved, allowing usage with CodaLab.
    # This must be done on both train.py and qa_answer.py in order to work.
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    os.symlink(os.path.abspath(FLAGS.train_dir), global_train_dir)
    train_dir = global_train_dir

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    logging.info(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    config = Config(question_len, context_len, 
                    FLAGS.max_gradient_norm, FLAGS.dropout, 
                    FLAGS.batch_size, FLAGS.epochs,
                    FLAGS.state_size, FLAGS.output_size,
                    FLAGS.embedding_size, FLAGS.learning_rate, train_dir)

    with tf.Graph().as_default():
        summary_writer = tf.summary.FileWriter(FLAGS.train_summary_dir)

        qa = QASystem(config, glove_vectors, rev_vocab, summary_writer)

        with tf.Session() as session:
            summary_writer.add_graph(session.graph)

            initialize_model(session, qa, train_dir)

            train_dataset = (train_question_ids, train_context_ids, train_answer_ids, train_question_masks, train_context_masks)
            val_dataset = (val_question_ids, val_context_ids, val_answer_ids, val_question_masks, val_context_masks)

            best_score = 0.

            for epoch in range(FLAGS.epochs):
                msg = "Epoch {} of {}:".format(epoch, FLAGS.epochs)
                print(msg)
                logging.info(msg)

                qa.train(session, train_dataset)
                score, em = qa.evaluate_answer(session, val_dataset, log=True)
                
                msg = "F1 score: {}, EM score: {}".format(score, em)
                print(msg)
                logging.info(msg)

                if score >= best_score:
                    best_score = score
                    qa.save_checkpoint(session)


if __name__ == "__main__":
    tf.app.run()







