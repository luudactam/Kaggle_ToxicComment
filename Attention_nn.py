from keras import backend as K
# from keras import initializations
from keras import initializers, regularizers, constraints
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.engine.topology import Layer
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Bidirectional, Conv1D, MaxPool1D, Flatten, GRU, Concatenate, SpatialDropout1D, GlobalMaxPool1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import Adam, RMSprop

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
from sklearn.model_selection import KFold,StratifiedKFold
import pandas as pd
import numpy as np




from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import gensim.models.keyedvectors as word2vec

class WordParse:
    def __init__(self, TRAIN_DATA_FILE, TEST_DATA_FILE, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
        self.special_character_removal = re.compile(r'[^a-z\d ]', re.IGNORECASE)

        self.replace_numbers = re.compile(r'\d+', re.IGNORECASE)

        self.TRAIN_DATA_FILE = TRAIN_DATA_FILE
        self.TEST_DATA_FILE = TEST_DATA_FILE
        self.MAX_NB_WORDS = MAX_NB_WORDS
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.EMBEDDING_DIM = EMBEDDING_DIM

        self.list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

        return

    def get_train_test(self):
        train_df = pd.read_csv(self.TRAIN_DATA_FILE)
        test_df = pd.read_csv(self.TEST_DATA_FILE)

        list_sentences_train = train_df["comment_text"].fillna("NA").values
        list_sentences_test = test_df["comment_text"].fillna("NA").values

        train_y = train_df[self.list_classes].values

        print('Processing Training Data...')
        comments_train = self.__get_word_list(list_sentences_train)

        print('Processing Testing Data...')
        comments_test = self.__get_word_list(list_sentences_test)

        train_data, test_data, self.word_index = self.__get_word_keys(comments_train, comments_test)

        return train_data, test_data, train_y

    def get_embedding_matrix(self, all_embeddings_index, all_embeddings_dim, all_embedding_type):

        print('Preparing embedding matrix')
        nb_words = min(self.MAX_NB_WORDS, len(self.word_index))
        embedding_matrix = np.zeros((nb_words, self.EMBEDDING_DIM))
        dim_begin = 0
        for emb_idx, embeddings_index in enumerate(all_embeddings_index):
            embedding_dim = dim_begin + all_embeddings_dim[emb_idx]
            print(embedding_dim)
            for word, i in self.word_index.items():
                if i >= self.MAX_NB_WORDS:
                    continue
                if all_embedding_type[emb_idx] == 'word2vec':
                    embedding_vector = embeddings_index.get(word)
                else:
                    embedding_vector = embeddings_index.get(str.encode(word, 'utf-8'))
                #
                # print(embedding_vector)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i, dim_begin:embedding_dim] = embedding_vector

            dim_begin = embedding_dim

        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

        return embedding_matrix

    def __get_word_keys(self, train_comments, test_comments):
        tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS)
        tokenizer.fit_on_texts(train_comments + test_comments)

        sequences = tokenizer.texts_to_sequences(train_comments)
        test_sequences = tokenizer.texts_to_sequences(test_comments)

        word_index = tokenizer.word_index
        print('Found %s unique tokens' % len(word_index))

        train_data = pad_sequences(sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        print('Shape of data tensor:', train_data.shape)

        test_data = pad_sequences(test_sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        print('Shape of test_data tensor:', test_data.shape)

        return train_data, test_data, word_index

    def __get_word_list(self, sentences):
        comments = []
        for ii, text in enumerate(sentences):
            if ii % 100000 == 0:
                print(str(ii) + ' samples...')
            comments.append(self.__text_to_wordlist(text))
        return comments

    def word2vec(self, EMBEDDING_FILE, EMBEDDING_TYPE):
        print('Indexing word vectors')
        # word vectors
        embeddings_index = {}
        if EMBEDDING_TYPE == 'glove' or EMBEDDING_TYPE == 'fasttext':
            f = open(EMBEDDING_FILE, 'rb')
            for ii, line in enumerate(f):
                if ii % 100000 == 0:
                    print(str(ii) + ' words ...')
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()
        elif EMBEDDING_TYPE == 'word2vec':
            word2vecDict = word2vec.KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
            for ii, word in enumerate(word2vecDict.vocab):
                if ii % 100000 == 0:
                    print(str(ii) + ' words ...')
                embeddings_index[word] = word2vecDict.word_vec(word)
        else:
            print('No such kind of word vectors!')
        print('Total %s word vectors.' % len(embeddings_index))

        return embeddings_index

    def __text_to_wordlist(self, text, remove_stopwords=False, stem_words=False):
        # Clean the text, with the option to remove stopwords and to stem words.

        # Convert words to lower case and split them
        text = text.lower().split()

        # Optionally, remove stop words
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]

        text = " ".join(text)

        # Remove Special Characters
        text = self.special_character_removal.sub('', text)

        # Replace Numbers
        text = self.replace_numbers.sub('n', text)

        # Optionally, shorten words to their stems
        if stem_words:
            text = text.split()
            stemmer = SnowballStemmer('english')
            stemmed_words = [stemmer.stem(word) for word in text]
            text = " ".join(stemmed_words)

        # Return a list of words
        return (text)


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim

class multi_nlp_model():
    def __init__(self, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST, EPOCHS,
                 EMBEDDING_DIM, OUTPUT_SIZE, OUTPUT_FEATURE_NUM, DENSE_HIDDEN_NUM, MAX_SEQ_LEN,
                 DROP_OUT_RATE_LSTM, DROP_OUT_RATE_DENSE,
                 MAX_NB_WORDS,
                 MODEL_PATH):
        self.batch_size = BATCH_SIZE_TRAIN
        self.batch_size_test = BATCH_SIZE_TEST
        self.epochs = EPOCHS
        self.embedding_dim = EMBEDDING_DIM
        self.output_size = OUTPUT_SIZE
        self.output_feature_num = OUTPUT_FEATURE_NUM
        self.dense_hidden_num = DENSE_HIDDEN_NUM
        self.max_seq_len = MAX_SEQ_LEN
        self.dropout_rate_lstm = DROP_OUT_RATE_LSTM
        self.dropout_rate_dense = DROP_OUT_RATE_DENSE
        self.word_index_num = MAX_NB_WORDS
        self.model_path = MODEL_PATH

    def get_bidirectional_multi_lstm_att(self, embedding_matrix, model_idx):
        comment_input = Input(shape=(self.max_seq_len,))
        lstm_out = []
        for emb_mat_tmp in embedding_matrix:
            embedding_layer = (Embedding(self.word_index_num, self.embedding_dim,
                                    weights=[emb_mat_tmp],
                                    input_length=self.max_seq_len, trainable=False))
            lstm_layer = (Bidirectional(LSTM(self.output_feature_num,
                          dropout=self.dropout_rate_lstm,
                          recurrent_dropout=self.dropout_rate_lstm,
                          return_sequences=True)))
            embedded_sequences = embedding_layer(comment_input)
            lstm_out.append(lstm_layer(embedded_sequences))
        x = 0
        for ii, out_tmp in enumerate(lstm_out):
            if ii == 0:
                x = out_tmp
            else:
                x = Concatenate()([x, out_tmp])
        x = Dropout(self.dropout_rate_dense)(x)
        merged = Attention(self.max_seq_len)(x)
        merged = Dense(self.dense_hidden_num, activation='relu')(merged)
        merged = Dropout(self.dropout_rate_dense)(merged)
        merged = BatchNormalization()(merged)
        y = Dense(self.output_size, activation='sigmoid')(merged)

        model = Model(inputs=[comment_input], outputs=y)
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        print(model.summary())

        model_tag = 'bidirectional_multi_lstm_attention_glove_vectors_drop_params_%.2f_%.2f' % (
        self.dropout_rate_lstm, self.dropout_rate_dense)
        self.bst_model_path = self.model_path + model_tag + '_model_idx_' + model_idx + '.h5'
        print(model_tag)
        self.reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model_checkpoint = ModelCheckpoint(self.bst_model_path, save_best_only=True, save_weights_only=True)
        self.model = model
        self.model_tag = model_tag

    def fit(self, data_train, labels_train, data_val, labels_val):
        print('training data size: ' + str(data_train.shape))
        print('training target size: ' + str(labels_train.shape))
        print('validation data size: ' + str(data_val.shape))
        print('validation target size: ' + str(labels_val.shape))
        self.hist = self.model.fit(data_train, labels_train,
                                   verbose=2,
                                   validation_data=(data_val, labels_val),
                                   epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                                   callbacks=[self.early_stopping, self.model_checkpoint, self.reduce_lr_on_plateau])

    def predict(self, data):
        print('testing data size: ' + str(data.shape))
        y = self.model.predict([data], batch_size=self.batch_size_test, verbose=2)
        return y

    def predict_from_saved_model(self, data, saved_model_path):
        y = 0
        print('testing data size: ' + str(data.shape))
        model_lists = os.listdir(saved_model_path)
        for ii, model_list in enumerate(model_lists):
            print('iteration ' + str(ii))
            model_tmp = load_model(saved_model_path + model_list)
            if ii == 0:
                y = model_tmp.predict([data], batch_size=self.batch_size_test, verbose=2)
            else:
                y = y + model_tmp.predict([data], batch_size=self.batch_size_test, verbose=2)
        return y





class nlp_model():
    def __init__(self, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST, EPOCHS,
                 EMBEDDING_DIM, OUTPUT_SIZE, OUTPUT_FEATURE_NUM, DENSE_HIDDEN_NUM, MAX_SEQ_LEN,
                 DROP_OUT_RATE_LSTM, DROP_OUT_RATE_DENSE,
                 MAX_NB_WORDS,
                 MODEL_PATH):
        self.batch_size = BATCH_SIZE_TRAIN
        self.batch_size_test = BATCH_SIZE_TEST
        self.epochs = EPOCHS
        self.embedding_dim = EMBEDDING_DIM
        self.output_size = OUTPUT_SIZE
        self.output_feature_num = OUTPUT_FEATURE_NUM
        self.dense_hidden_num = DENSE_HIDDEN_NUM
        self.max_seq_len = MAX_SEQ_LEN
        self.dropout_rate_lstm = DROP_OUT_RATE_LSTM
        self.dropout_rate_dense = DROP_OUT_RATE_DENSE
        self.word_index_num = MAX_NB_WORDS
        self.model_path = MODEL_PATH

    def get_bidirectional_gru_att(self, embedding_matrix, model_idx):
        embedding_layer = Embedding(self.word_index_num, self.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_seq_len, trainable=False)
        gru_layer = Bidirectional(GRU(self.output_feature_num,
                          dropout=self.dropout_rate_lstm,
                          recurrent_dropout=self.dropout_rate_lstm,
                          return_sequences=True))
        comment_input = Input(shape=(self.max_seq_len,))
        embedded_sequences = embedding_layer(comment_input)
        x = gru_layer(embedded_sequences)
        x = Dropout(self.dropout_rate_dense)(x)
        merged = Attention(self.max_seq_len)(x)
        merged = Dense(self.dense_hidden_num, activation='relu')(merged)
        merged = Dropout(self.dropout_rate_dense)(merged)
        merged = BatchNormalization()(merged)
        y = Dense(self.output_size, activation='sigmoid')(merged)

        model = Model(inputs=[comment_input], outputs=y)
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        print(model.summary())

        model_tag = 'bidirectional_gru_attention_glove_vectors_drop_params_%.2f_%.2f' % (
        self.dropout_rate_lstm, self.dropout_rate_dense)
        self.bst_model_path = self.model_path + model_tag + '_model_idx_' + str(model_idx) + '.h5'
        print(model_tag)
        self.reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model_checkpoint = ModelCheckpoint(self.bst_model_path, save_best_only=True, save_weights_only=True)
        self.model = model
        self.model_tag = model_tag

    def get_bidirectional_lstm_conv1d(self, embedding_matrix, model_idx):
        embedding_layer = Embedding(self.word_index_num, self.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_seq_len, trainable=False)
        lstm_layer = Bidirectional(LSTM(self.output_feature_num,
                                        dropout=self.dropout_rate_lstm,
                                        recurrent_dropout=self.dropout_rate_lstm,
                                        return_sequences=True))
        comment_input = Input(shape=(self.max_seq_len,))
        embedded_sequences = embedding_layer(comment_input)
        x = lstm_layer(embedded_sequences)
        x = Dropout(self.dropout_rate_dense)(x)
        feat_num = self.output_feature_num
        conv1 = Conv1D(feat_num, 3, padding='same')(x)
        conv1 = Conv1D(feat_num, 3, padding='same')(conv1)
        maxpl1 = MaxPool1D(padding='same')(conv1)
        norm1 = BatchNormalization()(maxpl1)
        conv2 = Conv1D(feat_num, 3, padding='same')(norm1)
        conv2 = Conv1D(feat_num, 3, padding='same')(conv2)
        maxpl2 = MaxPool1D(padding='same')(conv2)
        norm2 = BatchNormalization()(maxpl2)
        conv3 = Conv1D(feat_num, 3, padding='same')(norm2)
        conv3 = Conv1D(feat_num, 3, padding='same')(conv3)
        maxpl3 = MaxPool1D(padding='same')(conv3)
        norm3 = BatchNormalization()(maxpl3)
        conv4 = Conv1D(feat_num, 3, padding='same')(norm3)
        conv4 = Conv1D(feat_num, 3, padding='same')(conv4)
        maxpl4 = MaxPool1D(padding='same')(conv4)
        norm4 = BatchNormalization()(maxpl4)
        conv5 = Conv1D(feat_num, 3, padding='same')(norm4)
        conv5 = Conv1D(feat_num, 3, padding='same')(conv5)
        maxpl5 = MaxPool1D(padding='same')(conv5)
        norm5 = BatchNormalization()(maxpl5)

        merged = Flatten()(norm5)
        merged = Dropout(self.dropout_rate_dense)(merged)
        merged = Dense(self.dense_hidden_num, activation='relu')(merged)
        merged = Dropout(self.dropout_rate_dense)(merged)
        merged = BatchNormalization()(merged)

        y = Dense(self.output_size, activation='sigmoid')(merged)

        model = Model(inputs=[comment_input], outputs=y)
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        print(model.summary())

        model_tag = 'bidirectional_lstm_conv1d_glove_vectors_drop_params_%.2f_%.2f' % (
            self.dropout_rate_lstm, self.dropout_rate_dense)
        self.bst_model_path = self.model_path + model_tag + '_model_idx_' + str(model_idx) + '.h5'
        print(model_tag)
        self.reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model_checkpoint = ModelCheckpoint(self.bst_model_path, save_best_only=True, save_weights_only=True)
        self.model = model
        self.model_tag = model_tag


    def get_conv1d(self, embedding_matrix, model_idx):
        embedding_layer = Embedding(self.word_index_num, self.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_seq_len, trainable=False)
        comment_input = Input(shape=(self.max_seq_len,))
        embedded_sequences = embedding_layer(comment_input)
        feat_num = self.output_feature_num
        conv1 = Conv1D(feat_num, 3, padding='same')(embedded_sequences)
        conv1 = Conv1D(feat_num, 3, padding='same')(conv1)
        maxpl1 = MaxPool1D(padding='same')(conv1)
        norm1 = BatchNormalization()(maxpl1)
        conv2 = Conv1D(feat_num, 3, padding='same')(norm1)
        conv2 = Conv1D(feat_num, 3, padding='same')(conv2)
        maxpl2 = MaxPool1D(padding='same')(conv2)
        norm2 = BatchNormalization()(maxpl2)
        conv3 = Conv1D(feat_num, 3, padding='same')(norm2)
        conv3 = Conv1D(feat_num, 3, padding='same')(conv3)
        maxpl3 = MaxPool1D(padding='same')(conv3)
        norm3 = BatchNormalization()(maxpl3)
        conv4 = Conv1D(feat_num, 3, padding='same')(norm3)
        conv4 = Conv1D(feat_num, 3, padding='same')(conv4)
        maxpl4 = MaxPool1D(padding='same')(conv4)
        norm4 = BatchNormalization()(maxpl4)
        conv5 = Conv1D(feat_num, 3, padding='same')(norm4)
        conv5 = Conv1D(feat_num, 3, padding='same')(conv5)
        maxpl5 = MaxPool1D(padding='same')(conv5)
        norm5 = BatchNormalization()(maxpl5)

        merged = Flatten()(norm5)
        merged = Dropout(self.dropout_rate_dense)(merged)
        merged = Dense(self.dense_hidden_num, activation='relu')(merged)
        merged = Dropout(self.dropout_rate_dense)(merged)
        merged = BatchNormalization()(merged)

        y = Dense(self.output_size, activation='sigmoid')(merged)

        model = Model(inputs=[comment_input], outputs=y)
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        print(model.summary())

        model_tag = 'conv1d_glove_vectors_drop_params_%.2f_%.2f' % (
            self.dropout_rate_lstm, self.dropout_rate_dense)
        self.bst_model_path = self.model_path + model_tag + '_model_idx_' + str(model_idx) + '.h5'
        print(model_tag)
        self.reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model_checkpoint = ModelCheckpoint(self.bst_model_path, save_best_only=True, save_weights_only=True)
        self.model = model
        self.model_tag = model_tag

    def get_bidirectional_lstm_glbm(self, embedding_matrix, model_idx):
        embedding_layer = Embedding(self.word_index_num, self.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_seq_len, trainable=False)
        lstm_layer = Bidirectional(LSTM(self.output_feature_num,
                          dropout=self.dropout_rate_lstm,
                          recurrent_dropout=self.dropout_rate_lstm,
                          return_sequences=True))
        comment_input = Input(shape=(self.max_seq_len,))
        embedded_sequences = embedding_layer(comment_input)
        # x = SpatialDropout1D(self.dropout_rate_dense)(embedded_sequences)
        x = lstm_layer(embedded_sequences)
        x = Dropout(self.dropout_rate_dense)(x)
        global_max = GlobalMaxPool1D()(x)
        merged = Dense(self.dense_hidden_num, activation='relu')(global_max)
        merged = Dropout(self.dropout_rate_dense)(merged)
        merged = BatchNormalization()(merged)
        y = Dense(self.output_size, activation='sigmoid')(merged)

        model = Model(inputs=[comment_input], outputs=y)
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        print(model.summary())

        model_tag = 'bidirectional_lstm_attention_glove_vectors_drop_params_%.2f_%.2f' % (
        self.dropout_rate_lstm, self.dropout_rate_dense)
        self.bst_model_path = self.model_path + model_tag + '_model_idx_' + model_idx + '.h5'
        print(model_tag)
        self.reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model_checkpoint = ModelCheckpoint(self.bst_model_path, save_best_only=True, save_weights_only=True)
        self.model = model
        self.model_tag = model_tag


    def get_bidirectional_lstm_att(self, embedding_matrix, model_idx):
        embedding_layer = Embedding(self.word_index_num, self.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_seq_len, trainable=False)
        lstm_layer = Bidirectional(LSTM(self.output_feature_num,
                          dropout=self.dropout_rate_lstm,
                          recurrent_dropout=self.dropout_rate_lstm,
                          return_sequences=True))
        comment_input = Input(shape=(self.max_seq_len,))
        embedded_sequences = embedding_layer(comment_input)
        # x = SpatialDropout1D(self.dropout_rate_dense)(embedded_sequences)
        x = lstm_layer(embedded_sequences)
        x = Dropout(self.dropout_rate_dense)(x)
        merged = Attention(self.max_seq_len)(x)
        merged = Dense(self.dense_hidden_num, activation='relu')(merged)
        merged = Dropout(self.dropout_rate_dense)(merged)
        merged = BatchNormalization()(merged)
        y = Dense(self.output_size, activation='sigmoid')(merged)

        model = Model(inputs=[comment_input], outputs=y)
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        print(model.summary())

        model_tag = 'bidirectional_lstm_attention_glove_vectors_drop_params_%.2f_%.2f' % (
        self.dropout_rate_lstm, self.dropout_rate_dense)
        self.bst_model_path = self.model_path + model_tag + '_model_idx_' + model_idx + '.h5'
        print(model_tag)
        self.reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model_checkpoint = ModelCheckpoint(self.bst_model_path, save_best_only=True, save_weights_only=True)
        self.model = model
        self.model_tag = model_tag

    def get_model_lstm_att(self, embedding_matrix, model_idx):
        embedding_layer = Embedding(self.word_index_num, self.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_seq_len, trainable=False)
        lstm_layer = LSTM(self.output_feature_num,
                          dropout=self.dropout_rate_lstm,
                          recurrent_dropout=self.dropout_rate_lstm,
                          return_sequences=True)
        comment_input = Input(shape=(self.max_seq_len,))
        embedded_sequences = embedding_layer(comment_input)
        x = lstm_layer(embedded_sequences)
        x = Dropout(self.dropout_rate_dense)(x)
        merged = Attention(self.max_seq_len)(x)
        merged = Dense(self.dense_hidden_num, activation='relu')(merged)
        merged = Dropout(self.dropout_rate_dense)(merged)
        merged = BatchNormalization()(merged)
        y = Dense(self.output_size, activation='sigmoid')(merged)

        model = Model(inputs=[comment_input], outputs=y)
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        print(model.summary())

        model_tag = 'lstm_attention_glove_vectors_drop_params_%.2f_%.2f' % (self.dropout_rate_lstm, self.dropout_rate_dense)
        self.bst_model_path = self.model_path + model_tag + '_model_idx_' + str(model_idx) + '.h5'
        print(model_tag)
        self.reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model_checkpoint = ModelCheckpoint(self.bst_model_path, save_best_only=True, save_weights_only=True)
        self.model = model
        self.model_tag = model_tag

    def fit(self, data_train, labels_train, data_val, labels_val):
        print('training data size: ' + str(data_train.shape))
        print('training target size: ' + str(labels_train.shape))
        print('validation data size: ' + str(data_val.shape))
        print('validation target size: ' + str(labels_val.shape))
        self.hist = self.model.fit(data_train, labels_train,
                       verbose=2,
                       validation_data=(data_val, labels_val),
                       epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                       callbacks=[self.early_stopping, self.model_checkpoint, self.reduce_lr_on_plateau])

    def predict(self, data):
        print('testing data size: ' + str(data.shape))
        y = self.model.predict([data], batch_size=self.batch_size_test, verbose=2)
        return y

    def load_model(self, model = None):
        if model == None:
            self.model.load_weights(self.bst_model_path)
        else:
            self.model.load_weights(model)

    def predict_from_saved_model(self, data, saved_model_path):
        y = 0
        print('testing data size: ' + str(data.shape))
        model_lists = os.listdir(saved_model_path)
        print(str(len(model_lists)) + ' models loaded...')
        for ii, model_list in enumerate(model_lists):
            print('iteration '+str(ii))
            self.model.load_weights(saved_model_path+model_list)
            print('model '+model_list+' loaded')
            if ii == 0:
                y = self.model.predict([data], batch_size=self.batch_size_test, verbose=2)
            else:
                y = y + self.model.predict([data], batch_size=self.batch_size_test, verbose=2)
        y = y / len(model_lists)
        return y

def make_submission(SUBMISSION_FILE, y_test, submisstion_tag):
    # Make submission file & log file to tracing back
    sample_submission = pd.read_csv(SUBMISSION_FILE)
    sample_submission[word_parse.list_classes] = y_test
    sample_submission.to_csv(submission_path + str(EMBEDDING_TYPE) + '_' + nlp_model.model_tag + '_' + submisstion_tag +  '.csv', index=False)

if __name__ == '__main__':
    data_path = 'C:/Users/serig/Documents/Spyder/toxic/'
    embd_path = 'C:/Users/serig/Documents/Spyder/toxic/'
    model_path = 'C:/Users/serig/Documents/Spyder/toxic/'
    submission_path = 'C:/Users/serig/Documents/Spyder/toxic/'
    loc_path = 'C:/Users/serig/Documents/Spyder/toxic/'

    TRAIN_DATA_FILE = data_path + 'train.csv'
    TEST_DATA_FILE = data_path + 'test.csv'
    SUBMISSION_FILE = submission_path + 'sample_submission.csv'

    EMBEDDING_FILE_glove = embd_path + 'glove.840B.300d.txt'     # Standford GloVe word2vec database
    EMBEDDING_FILE_fasttext = embd_path + 'crawl-300d-2M.vec'     # Facebook Fasttext word2vec database
    EMBEDDING_FILE_word2vec = embd_path + 'GoogleNews-vectors-negative300.bin'  # Google word2vec database    

    EMBEDDING_TYPE_glove = 'glove'
    EMBEDDING_TYPE_fasttext = 'fasttext'
    EMBEDDING_TYPE_word2vec = 'word2vec'

    EMBEDDING_TYPE = [EMBEDDING_TYPE_fasttext]

    EMBEDDING_DIM_glove = 300  # feature number of word vector
    EMBEDDING_DIM_fasttext = 300
    EMBEDDING_DIM_word2vec = 300

    EMBEDDING_DIM = [EMBEDDING_DIM_fasttext]

    MAX_SEQUENCE_LENGTH = 150     # max sequence of each sentence
    MAX_NB_WORDS = 2000000         # max word quoted in GloVe

    VALIDATION_SPLIT = 0.1
    BATCH_SIZE_TRAIN = 128
    BATCH_SIZE_TEST = 128
    EPOCHS = 50
    OUTPUT_SIZE = 6
    BAGGING_K = 5

    OUTPUT_FEATURE_NUM = 300  # LSTM output features
    DENSE_HIDDEN_NUM = 256               # Dense hidden layer neurals
    DROP_OUT_RATE_LSTM = 0.25         # Dropout parameters
    DROP_OUT_RATE_DENSE = 0.25        # Dropout parameters

    word_parse = WordParse(TRAIN_DATA_FILE, TEST_DATA_FILE, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, sum(EMBEDDING_DIM))
    # embeddings_index_glove = word_parse.word2vec(EMBEDDING_FILE_glove, EMBEDDING_TYPE_glove)   # Get GloVe word vectors Index
    embeddings_index_fasttext = word_parse.word2vec(EMBEDDING_FILE_fasttext, EMBEDDING_TYPE_fasttext)   # Get Fasttext word vectors Index
    # embeddings_index_word2vec = word_parse.word2vec(EMBEDDING_FILE_word2vec, EMBEDDING_TYPE_word2vec)   # Get Word2vec word vectors Index

    embedding_index = [embeddings_index_fasttext]

    list_sentences_train, list_sentences_test, y_train =\
        word_parse.get_train_test()    # Get train and test input and train target



    embedding_matrix = word_parse.get_embedding_matrix(
        embedding_index,
        EMBEDDING_DIM,
        EMBEDDING_TYPE)  # Get embedding matrix by embedding index, this step has to be after encoded the all scentences, and has word_list information

    # del embeddings_index_glove
    # del embeddings_index_fasttext
    # del embeddings_index_word2vec

    print(min(MAX_NB_WORDS, len(word_parse.word_index)))

    nlp_model = \
        nlp_model(BATCH_SIZE_TRAIN, BATCH_SIZE_TEST, EPOCHS,
                  sum(EMBEDDING_DIM), OUTPUT_SIZE, OUTPUT_FEATURE_NUM, DENSE_HIDDEN_NUM, MAX_SEQUENCE_LENGTH,
                  DROP_OUT_RATE_LSTM, DROP_OUT_RATE_DENSE,
                  min(MAX_NB_WORDS, len(word_parse.word_index)), model_path)   # Initialized basic parameters

    kf = KFold(n_splits=BAGGING_K, shuffle=True)
    bagging_iter = 0
    y_test = 0
    for train_idx, val_idx in kf.split(list_sentences_train):
        print('BAGGING:' + str(bagging_iter) + ' PREDICTION,,,')
        data_tr = list_sentences_train[train_idx]
        y_tr = y_train[train_idx]
        data_vl = list_sentences_train[val_idx]
        y_vl = y_train[val_idx]

        nlp_model.get_bidirectional_lstm_att(embedding_matrix,
                                             str(EMBEDDING_TYPE) + '_' + str(bagging_iter))  # Get lstm attention model
        nlp_model.fit(data_tr, y_tr, data_vl, y_vl)  # Fit the model

        nlp_model.load_model()
        print(nlp_model.bst_model_path + ' model loaded')

        if bagging_iter == 0:
            y_test = nlp_model.predict(list_sentences_test)   # Get predictions
        else:
            y_test += nlp_model.predict(list_sentences_test)

        bagging_iter += 1

    y_test = y_test / BAGGING_K

    # Make submission file & log file to tracing back
    sample_submission = pd.read_csv(SUBMISSION_FILE)
    f = open(loc_path + 'history_' + nlp_model.model_tag + '.txt', 'w')
    f.write(str(nlp_model.hist.history))
    f.close()
    sample_submission[word_parse.list_classes] = y_test
    sample_submission.to_csv(submission_path + str(EMBEDDING_TYPE) + '_' + nlp_model.model_tag + '.csv', index=False)

