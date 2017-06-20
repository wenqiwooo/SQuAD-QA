import argparse
import numpy as np
from os.path import join as pjoin

import tensorflow as tf
from tensorflow.python.platform import gfile


class DataLoader:
	def __init__(self, datadir, question_filename, context_filename, 
								answer_filename, savedir, save_filename):
		self.datadir = datadir
		self.question_filename = question_filename
		self.context_filename = context_filename
		self.answer_filename = answer_filename
		self.savedir = savedir
		self.save_filename = save_filename

		self.question_filepath = pjoin(self.datadir, self.question_filename)
		self.context_filepath = pjoin(self.datadir, self.context_filename)
		self.answer_filepath = pjoin(self.datadir, self.answer_filename)
		self.save_filepath = pjoin(self.savedir, self.save_filename)

		self.question_len = None
		self.context_len = None

	def load(self, question_len=None, context_len=None):
		question_ids = []
		context_ids = []
		answer_ids = []
		question_masks = []
		context_masks = []

		with tf.gfile.GFile(self.question_filepath, mode='rb') as f:
			question_ids.extend(f.readlines())
			question_ids = [map(int, line.strip('\n').split()) for line in question_ids]

		with tf.gfile.GFile(self.context_filepath, mode='rb') as f:
			context_ids.extend(f.readlines())
			context_ids = [map(int, line.strip('\n').split()) for line in context_ids]

		with tf.gfile.GFile(self.answer_filepath, mode='rb') as f:
			answer_ids.extend(f.readlines())
			answer_ids = [map(int, line.strip('\n').split()) for line in answer_ids]

		if question_len is None:
			self.question_len = max(len(l) for l in question_ids)
		else:
			self.question_len = question_len

		if context_len is None:
			self.context_len = max(len(l) for l in context_ids)
		else:
			self.context_len = context_len

		data = zip(question_ids, context_ids, answer_ids)

		for d in data:
			q_ids, c_ids, a_ids = d
			if len(q_ids) > self.question_len or len(c_ids) > self.context_len:
				data.remove(d)
			else:
				question_masks.append(len(q_ids))
				context_masks.append(len(c_ids))
				q_ids += (self.question_len - len(q_ids)) * [0]
				c_ids += (self.context_len - len(c_ids)) * [0]

		question_ids, context_ids, answer_ids = zip(*data)
		question_ids = np.array(question_ids)
		context_ids = np.array(context_ids)
		answer_ids = np.array(answer_ids)
		question_masks = np.array(question_masks)
		context_masks = np.array(context_masks)

		print("Question ids shape: {}".format(question_ids.shape))	
		print("Context ids shape: {}".format(context_ids.shape))
		print("Answer ids shape: {}".format(answer_ids.shape))
		print("Question masks shape: {}".format(question_masks.shape))	
		print("Context masks shape: {}".format(context_masks.shape))

		return question_ids, context_ids, answer_ids, question_masks, context_masks

	def save(self, question_ids, context_ids, answer_ids, question_masks, context_masks):
		logging.info('writing to file %s', self.save_filepath)
		np.savez_compressed(self.save_filepath, 
												question_ids=question_ids, 
												context_ids=context_ids, 
												answer_ids=answer_ids,
												question_masks=question_masks,
												context_masks=context_masks)
		logging.info('saved training data as np.array at %s', self.save_filepath)

	def load_and_save(self, question_len=None, context_len=None):
		question_ids, context_ids, answer_ids, question_masks, context_masks = self.load(question_len=question_len, context_len=context_len)
		self.save(question_ids, context_ids, answer_ids, question_masks, context_masks)


def do_save(args):
	dl = DataLoader(args.data_dir, 
									args.q_filename, 
									args.c_filename, 
									args.a_filename, 
									args.save_dir, 
									args.s_filename)
	dl.load_and_save()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()

	command_parser = subparsers.add_parser('save', help='')
	command_parser.add_argument('-dd', '--data-dir', default="data/squad", help="Data directory")
	command_parser.add_argument('-sd', '--save-dir', default="data/squad", help="Save directory")
	command_parser.add_argument('-qf', '--q-filename', 
		default="train.ids.question", help="Question ids filename")
	command_parser.add_argument('-cf', '--c-filename', 
		default="train.ids.context", help="Context ids filename")
	command_parser.add_argument('-af', '--a-filename', 
		default="train.span", help="Answer ids filename")
	command_parser.add_argument('-sf', '--s-filename', 
		default="train.all", help="Saved data filename")
	command_parser.set_defaults(func=do_save)

	ARGS = parser.parse_args()
	if ARGS.func is None:
		parser.print_help()
		sys.exit(1)
	else:
		ARGS.func(ARGS)


