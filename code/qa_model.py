from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import numpy as np
from six.moves import xrange  
from collections import namedtuple
from os.path import join as pjoin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from evaluate import exact_match_score, f1_score
from util import *

from encoder import Encoder
from decoder import Decoder


class Config(object):
    def __init__(self, question_len, context_len, max_grad_norm, 
                dropout, batch_size, epochs, state_size, 
                output_size, embedding_size, lr, train_dir):
        self.question_len = question_len
        self.context_len = context_len
        self.max_grad_norm = max_grad_norm
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.state_size = state_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.lr = lr
        self.train_dir = train_dir


class QASystem(object):
    def __init__(self, params, pretrained_embeddings, rev_vocab, summary_writer=None):
        self.params = params
        self.pretrained_embeddings = pretrained_embeddings
        self.rev_vocab = rev_vocab
        self.summary_writer = summary_writer
        self.build()
        self.summary = tf.summary.merge_all()

    def build(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.add_placeholders()
        self.encoder = Encoder(self.params)
        self.decoder = Decoder(self.params)
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_train_op(self.loss)
        self.saver = tf.train.Saver()

    def add_placeholders(self):
        self.question_placeholder = tf.placeholder(tf.int32, [None, self.params.question_len], name="question")
        self.context_placeholder = tf.placeholder(tf.int32, [None, self.params.context_len], name="context")
        self.answer_placeholder = tf.placeholder(tf.int32, [None, 2], name="answer")
        self.question_mask_placeholder = tf.placeholder(tf.int32, [None, ], name="question_mask")
        self.context_mask_placeholder = tf.placeholder(tf.int32, [None, ], name="context_mask")
        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout")

    def create_feed_dict(self, question_batch, context_batch, question_masks, context_masks,
        answer_batch=None, dropout=1.0):
        
        feed_dict = {
            self.question_placeholder: question_batch,
            self.context_placeholder: context_batch,
            self.question_mask_placeholder: question_masks,
            self.context_mask_placeholder: context_masks,
            self.dropout_placeholder: dropout,
        }

        if answer_batch is not None:
            feed_dict[self.answer_placeholder] = answer_batch
        
        return feed_dict

    def add_embedding(self):
        embeddings = tf.Variable(self.pretrained_embeddings, trainable=False)
        question_embedding = tf.nn.embedding_lookup(embeddings, self.question_placeholder)
        context_embedding = tf.nn.embedding_lookup(embeddings, self.context_placeholder)

        question_embedding = tf.reshape(question_embedding, [-1, self.params.question_len, self.params.embedding_size])
        context_embedding = tf.reshape(context_embedding, [-1, self.params.context_len, self.params.embedding_size])

        return question_embedding, context_embedding

    def add_prediction_op(self):
        question_embedding, context_embedding = self.add_embedding()
        # U: (batch_size, context_len, 2*state_size)
        U = self.encoder.encode(question_embedding, context_embedding, 
            self.question_mask_placeholder, self.context_mask_placeholder, dropout=self.dropout_placeholder)
        preds = self.decoder.decode(U, self.context_mask_placeholder, dropout=self.dropout_placeholder)
        return preds

    def add_loss_op(self, preds):
        alphas, betas, _, _ = preds

        # labels_a, labels_b: (batch_size, 1)
        labels_s, labels_e = tf.split(1, 2, self.answer_placeholder)
        labels_s = tf.reshape(labels_s, [-1])
        labels_e = tf.reshape(labels_e, [-1])

        loss_s = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=a, labels=labels_s)) for a in alphas]
        loss_s = reduce(lambda x, y: x + y, loss_s) / len(loss_s)
        loss_e = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=b, labels=labels_e)) for b in betas]
        loss_e = reduce(lambda x, y: x + y, loss_e) / len(loss_e)

        loss = loss_s + loss_e
        tf.summary.scalar('loss', loss)
        return loss

    def add_train_op(self, loss):
        lr = self.params.lr
        lr = tf.train.exponential_decay(self.params.lr, self.global_step, 200, 0.97)
        lr = tf.maximum(0.000001, lr)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.params.max_grad_norm)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        return train_op

    def train(self, session, dataset, best_score=None):
        epoch_loss = 0.0
        duration = int(len(dataset[0]) / self.params.batch_size) + 1
        prog = Progbar(target=duration)

        for i, batch in enumerate(get_minibatches(dataset, self.params.batch_size)):
            question_batch, context_batch, answer_batch, question_masks, context_masks = batch
            loss = self.optimize(session, (question_batch, context_batch), 
                answer_batch, (question_masks, context_masks))
            epoch_loss += loss
            #prog.update(i + 1, [("train loss", loss)])
            print("{}/{} train loss: {}".format(i, duration, loss))

            if i % 500 == 0:
                self.save_checkpoint(session)
                score, em = self.evaluate_answer(session, dataset)
                print("F1 score: {}, EM score: {}".format(score, em))

        return epoch_loss

    def optimize(self, session, train_x, train_y, masks):
        question_batch, context_batch = train_x
        answer_batch = train_y
        question_masks, context_masks = masks

        input_feed = self.create_feed_dict(question_batch, context_batch, question_masks, context_masks, 
            answer_batch=answer_batch, dropout=self.params.dropout)
        output_feed = [self.train_op, self.loss, self.pred, self.summary]

        _, loss, preds, summary = session.run(output_feed, feed_dict=input_feed)

        if self.summary_writer is not None:
            self.summary_writer.add_summary(summary)

        _, _, slist, elist = preds
        print(slist[-1])
        print(elist[-1])
        return loss

    def test(self, session, valid_x, valid_y, masks):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        question_batch, context_batch = valid_x
        answer_batch = valid_y
        question_masks, context_masks = masks

        input_feed = self.create_feed_dict(question_batch, context_batch, question_masks, context_masks, 
            answer_batch=answer_batch, dropout=self.params.dropout)
        output_feed = [self.loss]

        loss = session.run(output_feed, input_feed)
        return loss

    def decode(self, session, test_x, masks):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        question_batch, context_batch = test_x
        question_masks, context_masks = masks
        input_feed = self.create_feed_dict(question_batch, context_batch, question_masks, context_masks, 
            dropout=self.params.dropout)
        output_feed = [self.pred]

        preds = session.run(output_feed, input_feed)
        return preds

    def answer(self, session, test_x, masks):
        preds = self.decode(session, test_x, masks)
        # answer should be list of (start, end) points
        _, _, slist, elist = preds[0]
        s = slist[-1]
        e = elist[-1]
        ans = zip(s.tolist(), e.tolist())
        return ans

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0
        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)
        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """
        question_batch, context_batch, answer_batch, question_masks, context_masks = dataset
        pred_spans = self.answer(session, 
            (question_batch[:sample,:], context_batch[:sample,:]), 
            (question_masks[:sample], context_masks[:sample]))

        predictions = batch_span_to_sentence(self.rev_vocab, pred_spans, context_batch)
        ground_truths = batch_span_to_sentence(self.rev_vocab, answer_batch, context_batch)
        eval_input = zip(predictions, ground_truths)

        f1 = [f1_score(p, t) for p, t in eval_input]
        f1 = reduce(lambda x, y: x + y, f1) / len(f1)
        em = [exact_match_score(p, t) for p, t in eval_input]
        em = reduce(lambda x, y: x + y, em) / len(em)

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def generate_answers(self, session, question_batch, context_batch, 
        question_masks, context_masks):
        print("Generating answers...")
        answers = []

        for i, batch in enumerate(get_validation_minibatches(question_batch, 
            context_batch, question_masks, context_masks, self.params.batch_size)):

            question_ids, context_ids, question_masks, context_masks = batch
            ans = self.answer(session, (question_ids, context_ids), (question_masks, context_masks))
            answers += ans
            print("Processed batch {}".format(i))

        return answers

    def save_checkpoint(self, session, global_step=None, name=None):
        if name is None:
            name = "qa-model"
        if global_step is None:
            global_step = self.global_step
        self.saver.save(session, pjoin(self.params.train_dir, name), global_step=global_step)










