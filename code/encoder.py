import tensorflow as tf
import qa_model


class Encoder(object):
    def __init__(self, params):
        self.params = params

    def encode(self, question_embedding, context_embedding, question_mask, context_mask, dropout=1.0):
        return self.co_attn_encode(question_embedding, context_embedding, question_mask, context_mask, dropout=dropout)

    def co_attn_encode(self, question_embedding, context_embedding, question_mask, context_mask, dropout=1.0):
        """
        question_embedding: (batch_size, question_len, vocab_dim)
        context_embedding: (batch_size, context_len, vocab_dim)
        """
        enc_lstm = tf.nn.rnn_cell.LSTMCell(self.params.state_size)

        f = lambda x: tf.concat(0, [x, tf.zeros([1, self.params.state_size], dtype=tf.float32)])
        
        with tf.variable_scope('enc_question'):
            # q_outputs: (batch_size, question_len, state_size)
            q_outputs, _ = tf.nn.dynamic_rnn(enc_lstm, question_embedding, dtype=tf.float32, sequence_length=question_mask)

            # add sentinel vector of dimension (1, state_size)
            # q: (batch_size, question_len+1, state_size)
            q = tf.map_fn(lambda x: f(x), q_outputs, dtype=tf.float32)

            # W_q: (state_size, state_size)
            W_q = tf.get_variable('W_q', [self.params.state_size , self.params.state_size], 
                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            # b_q: (question_len+1, state_size)
            b_q = tf.get_variable('b_q', [self.params.question_len+1, self.params.state_size], 
                dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            # q: (batch_size, question_len+1, state_size)
            q = tf.map_fn(lambda x: tf.tanh(tf.matmul(x, W_q) + b_q), q, dtype=tf.float32)

            q = tf.nn.dropout(q, dropout)
            
            # q_t: (batch_size, state_size, question_len+1)
            q_t = tf.transpose(q, perm=[0, 2, 1])

        with tf.variable_scope('enc_context'):
            # c_outputs: (batch_size, context_len, state_size)
            c_outputs, _ = tf.nn.dynamic_rnn(enc_lstm, context_embedding, dtype=tf.float32, sequence_length=context_mask)
            
            # c: (batch_size, context_len+1, state_size)
            c = tf.map_fn(lambda x: f(x), c_outputs, dtype=tf.float32)

        with tf.variable_scope('enc_attention'):
            # L: (batch_size, context_len+1, question_len+1)
            L = tf.batch_matmul(c, q_t)
            # L_t: (batch_size, question_len+1, context_len+1)
            L_t = tf.transpose(L, perm=[0, 2, 1])

            # A_q: (batch_size, question_len+1, context_len+1)
            A_q = tf.map_fn(lambda x: tf.nn.softmax(x), L_t, dtype=tf.float32)
            # A_c: (batch_size, context_len+1, question_len+1)
            A_c = tf.map_fn(lambda x: tf.nn.softmax(x), L, dtype=tf.float32)
            
            # C_q: (batch_size, question_len+1, state_size)
            C_q = tf.batch_matmul(A_q, c)
            # [q_t; transpose(C_q)]: (batch_size, 2*state_size, question_len+1)
            # C_q: (batch_size, 2*state_size, context_len+1) 
            C_d = tf.batch_matmul(tf.concat(1, [q_t, tf.transpose(C_q, perm=[0, 2, 1])]), tf.transpose(A_c, perm=[0, 2, 1]))

            # attention: (batch_size, context_len+1, 3*state_size)
            attention = tf.concat(2, [c, tf.transpose(C_d, perm=[0, 2, 1])])

        with tf.variable_scope('enc_bi_lstm'):
            enc_lstm_fw = tf.nn.rnn_cell.LSTMCell(self.params.state_size)
            enc_lstm_bw = tf.nn.rnn_cell.LSTMCell(self.params.state_size)
            
            # outputs: tuple of (output_fw, output_bw)
            #max_seq = tf.map_fn(lambda x: self.params.context_len, context_mask, dtype=tf.int32)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(enc_lstm_fw, enc_lstm_bw, attention, 
                sequence_length=context_mask, dtype=tf.float32)
            # U: (batch_size, context_len+1, 2*state_size)
            U = tf.concat(2, outputs)
            U = U[:,:self.params.context_len,:]

        return U








