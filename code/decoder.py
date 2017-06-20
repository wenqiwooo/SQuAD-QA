import tensorflow as tf
import numpy as np
import qa_model


class Decoder(object):
    def __init__(self, params, decode_type="pointer"):
        self.params = params
        self.decode_type = decode_type

    def decode(self, U, context_mask=None, dropout=1.0):
        return self.dynamic_decode(U, context_mask, dropout=dropout)

    def dynamic_linear(self, u, h, u_s, u_e, dropout=1.0):
        """
        u: (batch_size, 2*state_size)
        u_s, u_e: (batch_size, 2*state_size)
        h: (batch_size, state_size)
        """
        W_D = tf.get_variable("W_D", [5*self.params.state_size, self.params.state_size], 
            dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        W_1 = tf.get_variable("W_1", [3*self.params.state_size, self.params.state_size], 
            dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b_1 = tf.get_variable("b_1", [self.params.state_size], 
            dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        W_2 = tf.get_variable("W_2", [self.params.state_size, self.params.state_size], 
            dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b_2 = tf.get_variable("b_2", [self.params.state_size], 
            dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        W_3 = tf.get_variable("W_3", [2*self.params.state_size, 1],
            dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b_3 = tf.get_variable("b_3", [1],
            dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        # r_input: (batch_size, 5*state_size)
        r_input = tf.concat(1, [h, u_s, u_e])
        # r: (batch_size, state_size)
        r = tf.tanh(tf.matmul(r_input, W_D))
        r = tf.nn.dropout(r, dropout)

        # m_1_input: (batch_size, 3*state_size)
        m_1_input = tf.concat(1, [u, r])
        # m_1: (batch_size, state_size)
        m_1 = tf.matmul(m_1_input, W_1) + b_1
        m_1 = tf.nn.dropout(m_1, dropout)

        # m_2: (batch_size, state_size)
        m_2 = tf.matmul(m_1, W_2) + b_2
        m_2 = tf.nn.dropout(m_2, dropout)

        # m_3_input: (batch_size, 2*state_size)
        m_3_input = tf.concat(1, [m_1, m_2])
        m_3 = tf.matmul(m_3_input, W_3) + b_3
        m_3 = tf.reshape(m_3, [-1])

        return m_3


    def dynamic_decode(self, U, context_mask, dropout=1.0):
        """
        U: (batch_size, context_len, 2*state_size)
        """
        slist = []
        elist = []
        alphas = []
        betas = []

        with tf.variable_scope('decoder'):
            # u_s, u_e: (batch_size, 2*state_size)
            u_s = U[:, 0, :]
            u_e = U[:, 0, :]
            u_s = tf.reshape(u_s, [-1, 2*self.params.state_size])
            u_e = tf.reshape(u_e, [-1, 2*self.params.state_size])

            dec_lstm = tf.nn.rnn_cell.LSTMCell(self.params.state_size)

            dec_h = tf.map_fn(lambda x: tf.zeros([self.params.state_size]), U, dtype=tf.float32)
            dec_c = tf.map_fn(lambda x: tf.zeros([self.params.state_size]), U, dtype=tf.float32)

            # U_t: (context_len, batch_size, 2*state_size)
            U_t = tf.transpose(U, perm=[1, 0, 2])

            # This is the slow part
            for step in range(3):
                if step > 0: tf.get_variable_scope().reuse_variables()

                # lstm_input: (batch_size, 4*state_size)
                lstm_input = tf.concat(1, [u_s, u_e])
                # lstm_state: tuple of (h, c)
                _, lstm_state = dec_lstm(lstm_input, (dec_h, dec_c))
                # dec_h, dec_c: (batch_size, state_size)
                dec_h, dec_c = lstm_state

                with tf.variable_scope('start'):
                    # alpha: (context_len, batch_size)
                    alpha = tf.map_fn(lambda x: self.dynamic_linear(x, dec_h, u_s, u_e, dropout=dropout), 
                        U_t, dtype=tf.float32)
                    alpha_t = tf.transpose(alpha, perm=[1,0])
                    # s: (batch_size)
                    s = tf.reshape(tf.argmax(alpha, 0), [-1])
                    s = tf.to_int32(s)
                    # u_s: (batch_size, batch_size, 2*state_size)
                    u_s = tf.map_fn(lambda x: U[:,x,:], s, dtype=tf.float32)
                    u_s = tf.transpose(u_s, perm=[2, 0, 1])
                    u_s = tf.transpose(tf.matrix_diag_part(u_s), perm=[1,0])

                with tf.variable_scope('end'):
                    # beta: (context_len, batch_size)
                    beta = tf.map_fn(lambda x: self.dynamic_linear(x, dec_h, u_s, u_e, dropout=dropout), 
                        U_t, dtype=tf.float32)
                    beta_t = tf.transpose(beta, perm=[1,0])
                    # e: (batch_size)
                    e = tf.reshape(tf.argmax(beta, 0), [-1])
                    e = tf.to_int32(e)
                    # u_e: (batch_size, batch_size, 2*state_size)
                    u_e = tf.map_fn(lambda x: U[:,x,:], e, dtype=tf.float32)
                    # u_e: (2*state_size, batch_size, batch_size)
                    u_e = tf.transpose(u_e, perm=[2, 0, 1])
                    # u_e: (batch_size, 2*state_size)
                    u_e = tf.transpose(tf.matrix_diag_part(u_e), perm=[1,0])

                slist.append(s)
                elist.append(e)
                alphas.append(alpha_t)
                betas.append(beta_t)

        return alphas, betas, slist, elist



