'''
Created on Oct 8, 2020
@author: Inwoo Chung (gutomitai@gmail.com)
'''

import os
import time
import json
import random
from random import shuffle
import ctypes

#ctypes.WinDLL('cudart64_110.dll') #?

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, Dense, Concatenate, Dropout
from tensorflow.keras.layers import LSTM, Bidirectional, BatchNormalization, LayerNormalization
from tensorflow.keras.layers import Embedding, Layer
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (TensorBoard
    , ReduceLROnPlateau
    , LearningRateScheduler
    , ModelCheckpoint
    , EarlyStopping)
from tensorflow.keras.constraints import UnitNorm
from tensorflow.keras.initializers import RandomUniform, TruncatedNormal
from tensorflow.keras import regularizers

from ku.composite_layer import DenseBatchNormalization
from ku.backprop import (make_decoder_from_encoder
    , make_autoencoder_from_encoder
    , make_autoencoder_with_sym_sc)

# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# Constants.
DEBUG = True

MODE_TRAIN = 0
MODE_VAL = 1

CV_TYPE_TRAIN_VAL_SPLIT = 'train_val_split'
CV_TYPE_K_FOLD = 'k_fold'

DATASET_TYPE_PLAIN = 'plain'
DATASET_TYPE_BALANCED = 'balanced'

LOSS_TYPE_MULTI_LABEL = 'multi_label'
LOSS_TYPE_ADDITIVE_ANGULAR_MARGIN = 'additive_angular_margin'

epsilon = 1e-7


class MoALoss(Loss):
    def __init__(self
                 , W
                 , m=0.5
                 , ls=0.2
                 , scale=64.0
                 , loss_type=LOSS_TYPE_ADDITIVE_ANGULAR_MARGIN
                 , name='MoA_loss'):
        super(MoALoss, self).__init__(name=name)
        self.W = W
        self.m = m
        self.ls = ls
        self.scale = scale
        self.loss_type = loss_type

    #@tf.function
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        pos_mask = y_true
        neg_mask = 1.0 - y_true

        # Label smoothing.
        y_true = pos_mask * y_true * (1.0 - self.ls / 2.0) + neg_mask * (y_true + self.ls / 2.0)

        '''
        pos_log_loss = pos_mask * self.W[:, :, 0] * tf.sqrt(tf.square(y_true - y_pred))
        pos_log_loss_mean = tf.reduce_mean(pos_log_loss, axis=0) #?
        pos_loss = 1.0 * tf.reduce_mean(pos_log_loss_mean, axis=0)

        neg_log_loss = neg_mask * self.W[:, :, 1] * tf.sqrt(tf.square(y_true - y_pred))
        neg_log_loss_mean = tf.reduce_mean(neg_log_loss, axis=0) #?
        neg_loss = 1.0 * tf.reduce_mean(neg_log_loss_mean, axis=0)

        loss = pos_loss + neg_loss
        '''

        '''
        loss = tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred)))
        loss = tf.losses.binary_crossentropy(y_true, y_pred)
        log_loss_mean = tf.reduce_mean(log_loss, axis=0) #?
        loss = tf.reduce_mean(log_loss_mean, axis=0)
        '''

        if self.loss_type == LOSS_TYPE_ADDITIVE_ANGULAR_MARGIN:
            A = y_pred
            e_AM_A = tf.math.exp(self.scale * tf.math.cos(tf.math.acos(A) + self.m))
            #d = A.shape[-1] #?
            S = tf.tile(tf.reduce_sum(tf.math.exp(A), axis=1, keepdims=True), (1, 206))
            S_p = S - tf.math.exp(A) + e_AM_A
            P = e_AM_A / (S_p + epsilon)
            #P = tf.clip_by_value(P, clip_value_min=epsilon, clip_value_max=(1.0 - epsilon))

            #log_loss_1 = -1.0 * self.W[:, :, 0] * y_true * tf.math.log(P)
            log_loss_1 = -1.0 * y_true * tf.math.log(P)
            log_loss_2 = tf.reduce_sum(log_loss_1, axis=1)
            loss = tf.reduce_mean(log_loss_2, axis=0)
        elif self.loss_type == LOSS_TYPE_MULTI_LABEL:
            y_pred = tf.sigmoid(y_pred)
            y_pred = tf.maximum(tf.minimum(y_pred, 1.0 - 1e-15), 1e-15)
            log_loss = -1.0 * (y_true * tf.math.log(y_pred + epsilon) + (1.0 - y_true) * tf.math.log(1.0 - y_pred + epsilon))
            log_loss_mean = tf.reduce_mean(log_loss, axis=0) #?
            loss = tf.reduce_mean(log_loss_mean, axis=0)
        else:
            raise ValueError('loss type is not valid.')

        #tf.print(A, e_AM_A, S, S_p, P, log_loss_1, log_loss_2, loss)
        return loss

    def get_config(self):
        """Get configuration."""
        config = {'W': self.W
                  , 'm': self.m
                  , 'ls': self.ls
                  , 'scale': self.scale
                  , 'loss_type': self.loss_type}
        base_config = super(MoALoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MoAMetric(Metric):
    def __init__(self, sn_t=2.45, name='MoA_metric', **kwargs):
        super(MoAMetric, self).__init__(name=name, **kwargs)
        self.sn_t = sn_t
        self.total_loss = self.add_weight(name='total_loss', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        E = tf.reduce_mean(tf.math.exp(y_pred), axis=1, keepdims=True)
        E_2 = tf.reduce_mean(tf.square(tf.math.exp(y_pred)), axis=1, keepdims=True)
        S = tf.sqrt(E_2 - tf.square(E))

        e_A = (tf.exp(y_pred) - E) / (S + epsilon)
        e_A_p = tf.where(tf.math.greater(e_A, self.sn_t), self.sn_t, 0.0)
        p_hat = e_A_p / (tf.reduce_sum(e_A_p, axis=1, keepdims=True) + epsilon)

        y_pred = tf.maximum(tf.minimum(p_hat, 1.0 - 1e-15), 1e-15)
        y_true = tf.cast(y_true, dtype=tf.float32)

        log_loss = -1.0 * (y_true * tf.math.log(y_pred + epsilon) + (1.0 - y_true) * tf.math.log(1.0 - y_pred + epsilon))
        log_loss_mean = tf.reduce_mean(log_loss, axis=0) #?
        loss = tf.reduce_mean(log_loss_mean, axis=0)

        self.total_loss.assign_add(loss)
        self.count.assign_add(tf.constant(1.0))

    def result(self):
        return tf.math.divide_no_nan(self.total_loss, self.count)

    def reset_states(self):
        self.total_loss.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        """Get configuration."""
        config = {'sn_t': self.sn_t}
        base_config = super(MoAMetric, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class _MoAPredictor(Layer):
    def __init__(self, conf, **kwargs):
        super(_MoAPredictor, self).__init__(**kwargs)

        # Initialize.
        self.conf = conf
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']

        # Design layers.
        # First layers.
        self.embed_treatment_type_0 = Embedding(self.nn_arch['num_treatment_type']
                                           , self.nn_arch['d_input_feature'])
        self.dense_treatment_type_0 = Dense(self.nn_arch['d_input_feature']
                                       , activation='relu')

        self.layer_normalization_0_1 = LayerNormalization()
        self.layer_normalization_0_2 = LayerNormalization()
        self.layer_normalization_0_3 = LayerNormalization()

        # Autoencoder for gene expression profile.
        input_gene_exp_1 = Input(shape=(self.nn_arch['d_gene_exp'],))
        d_geps = [int(self.nn_arch['d_gep_init'] / np.power(2, v)) for v in range(4)]

        dense_1_1 = Dense(d_geps[0], activation='swish', kernel_regularizer=regularizers.l2(self.hps['weight_decay']))
        batch_normalization_1_1 = BatchNormalization()
        dropout_1_1 = None # Dropout(self.nn_arch['dropout_rate'])
        dense_batch_normalization_1_1 = DenseBatchNormalization(dense_1_1
                                                                , batch_normalization_1_1
                                                                , dropout=dropout_1_1)

        dense_1_2 = Dense(d_geps[1], activation='swish', kernel_regularizer=regularizers.l2(self.hps['weight_decay']))
        batch_normalization_1_2 = BatchNormalization()
        dropout_1_2 = None # Dropout(self.nn_arch['dropout_rate'])
        dense_batch_normalization_1_2 = DenseBatchNormalization(dense_1_2
                                                                , batch_normalization_1_2
                                                                , dropout=dropout_1_2)

        dense_1_3 = Dense(d_geps[2], activation='swish', kernel_regularizer=regularizers.l2(self.hps['weight_decay']))
        batch_normalization_1_3 = BatchNormalization()
        dropout_1_3 = None #Dropout(self.nn_arch['dropout_rate'])
        dense_batch_normalization_1_3 = DenseBatchNormalization(dense_1_3
                                                                , batch_normalization_1_3
                                                                , dropout=dropout_1_3)

        dense_1_4 = Dense(d_geps[3], activation='swish', kernel_regularizer=regularizers.l2(self.hps['weight_decay']))
        batch_normalization_1_4 = BatchNormalization()
        dropout_1_4 = None #Dropout(self.nn_arch['dropout_rate'])
        dense_batch_normalization_1_4 = DenseBatchNormalization(dense_1_4
                                                                , batch_normalization_1_4
                                                                , dropout=dropout_1_4)

        self.encoder_gene_exp_1 = keras.Sequential([input_gene_exp_1
                                                    , dense_batch_normalization_1_1
                                                    , dense_batch_normalization_1_2
                                                    , dense_batch_normalization_1_3
                                                    , dense_batch_normalization_1_4])
        self.decoder_gene_exp_1 = make_decoder_from_encoder(self.encoder_gene_exp_1)
        self.dropout_1 = Dropout(self.nn_arch['dropout_rate'])

        # Autoencoder for cell type.
        input_gene_exp_2 = Input(shape=(self.nn_arch['d_cell_type'],))
        d_cvs = [int(self.nn_arch['d_cv_init'] / np.power(2, v)) for v in range(3)]

        dense_2_1 = Dense(d_cvs[0], activation='swish', kernel_regularizer=regularizers.l2(self.hps['weight_decay']))
        batch_normalization_2_1 = BatchNormalization()
        dropout_2_1 = None # Dropout(self.nn_arch['dropout_rate'])
        dense_batch_normalization_2_1 = DenseBatchNormalization(dense_2_1
                                                                , batch_normalization_2_1
                                                                , dropout=dropout_2_1)

        dense_2_2 = Dense(d_cvs[1], activation='swish', kernel_regularizer=regularizers.l2(self.hps['weight_decay']))
        batch_normalization_2_2 = BatchNormalization()
        dropout_2_2 = None # Dropout(self.nn_arch['dropout_rate'])
        dense_batch_normalization_2_2 = DenseBatchNormalization(dense_2_2
                                                                , batch_normalization_2_2
                                                                , dropout=dropout_2_2)

        dense_2_3 = Dense(d_cvs[2], activation='swish', kernel_regularizer=regularizers.l2(self.hps['weight_decay']))
        batch_normalization_2_3 = BatchNormalization()
        dropout_2_3 = None #Dropout(self.nn_arch['dropout_rate'])
        dense_batch_normalization_2_3 = DenseBatchNormalization(dense_2_3
                                                                , batch_normalization_2_3
                                                                , dropout=dropout_2_3)

        self.encoder_cell_type_2 = keras.Sequential([input_gene_exp_2
                                                    , dense_batch_normalization_2_1
                                                    , dense_batch_normalization_2_2
                                                    , dense_batch_normalization_2_3])
        self.decoder_cell_type_2 = make_decoder_from_encoder(self.encoder_cell_type_2)
        self.dropout_2 = Dropout(self.nn_arch['dropout_rate'])

        # Skip-connection autoencoder layer.
        self.sc_aes = []
        self.dropout_3 = Dropout(self.nn_arch['dropout_rate'])

        for i in range(self.nn_arch['num_sc_ae']):
            input_sk_ae_3 = Input(shape=(self.nn_arch['d_hidden'],))
            d_ae_init = d_geps[-1] + d_cvs[-1] + self.nn_arch['d_input_feature']
            d_aes = [d_ae_init, int(d_ae_init * 2), int(d_ae_init * 2), d_ae_init]

            dense_3_1 = Dense(d_aes[0], activation='swish', kernel_regularizer=regularizers.l2(self.hps['weight_decay']))
            batch_normalization_3_1 = BatchNormalization()
            dropout_3_1 = None # Dropout(self.nn_arch['dropout_rate'])
            dense_batch_normalization_3_1 = DenseBatchNormalization(dense_3_1
                                                                    , batch_normalization_3_1
                                                                    , dropout=dropout_3_1)

            dense_3_2 = Dense(d_aes[1], activation='swish', kernel_regularizer=regularizers.l2(self.hps['weight_decay']))
            batch_normalization_3_2 = BatchNormalization()
            dropout_3_2 = None # Dropout(self.nn_arch['dropout_rate'])
            dense_batch_normalization_3_2 = DenseBatchNormalization(dense_3_2
                                                                    , batch_normalization_3_2
                                                                    , dropout=dropout_3_2)

            dense_3_3 = Dense(d_aes[2], activation='swish', kernel_regularizer=regularizers.l2(self.hps['weight_decay']))
            batch_normalization_3_3 = BatchNormalization()
            dropout_3_3 = None # Dropout(self.nn_arch['dropout_rate'])
            dense_batch_normalization_3_3 = DenseBatchNormalization(dense_3_3
                                                                    , batch_normalization_3_3
                                                                    , dropout=dropout_3_3)

            dense_3_4 = Dense(d_aes[3], activation='swish', kernel_regularizer=regularizers.l2(self.hps['weight_decay']))
            batch_normalization_3_4 = BatchNormalization()
            dropout_3_4 = None # Dropout(self.nn_arch['dropout_rate'])
            dense_batch_normalization_3_4 = DenseBatchNormalization(dense_3_4
                                                                    , batch_normalization_3_4
                                                                    , dropout=dropout_3_4)

            sc_encoder_3 = keras.Sequential([input_sk_ae_3
                                                        , dense_batch_normalization_3_1
                                                        , dense_batch_normalization_3_2
                                                        , dense_batch_normalization_3_3
                                                        , dense_batch_normalization_3_4])
            sc_autoencoder_3 = make_autoencoder_from_encoder(sc_encoder_3)
            self.sc_aes.append(make_autoencoder_with_sym_sc(sc_autoencoder_3))

        # Final layers.
        d_fs = [int(self.nn_arch['d_f_init'] / np.power(2, v)) for v in range(3)]

        self.dense_4_1 = Dense(d_fs[0], activation='swish', kernel_regularizer=regularizers.l2(self.hps['weight_decay']))
        self.dense_4_2 = Dense(d_fs[1], activation='swish', kernel_regularizer=regularizers.l2(self.hps['weight_decay']))
        self.dense_4_3 = Dense(d_fs[2], activation='swish', kernel_regularizer=regularizers.l2(self.hps['weight_decay']))
        self.dropout_4_3 = Dropout(self.nn_arch['dropout_rate'])

        if self.conf['loss_type'] == LOSS_TYPE_MULTI_LABEL:
            self.dense_4_4 = Dense(self.nn_arch['num_moa_annotation']
                                   , activation='linear'
                                   , kernel_initializer=TruncatedNormal()
                                   , kernel_constraint=None
                                   , kernel_regularizer=regularizers.l2(self.hps['weight_decay'])
                                   , use_bias=False) #?
        elif self.conf['loss_type'] == LOSS_TYPE_ADDITIVE_ANGULAR_MARGIN:
            self.dense_4_4 = Dense(self.nn_arch['num_moa_annotation']
                                   , activation='linear'
                                   , kernel_initializer=TruncatedNormal()
                                   , kernel_constraint=UnitNorm()
                                   , kernel_regularizer=regularizers.l2(self.hps['weight_decay'])
                                   , use_bias=False) #?
        else:
            raise ValueError('loss type is not valid.')

    def call(self, inputs):
        t = inputs[0]
        g = inputs[1]
        c = inputs[2]

        # First layers.
        t = self.embed_treatment_type_0(t)
        t = tf.reshape(t, (-1, self.nn_arch['d_input_feature']))
        t = self.dense_treatment_type_0(t)

        t = self.layer_normalization_0_1(t)
        g = self.layer_normalization_0_2(g)
        c = self.layer_normalization_0_3(c)

        # Gene expression.
        g_e = self.encoder_gene_exp_1(g)
        x_g = self.decoder_gene_exp_1(g_e)
        x_g = tf.expand_dims(x_g, axis=-1)
        x_g = tf.squeeze(x_g, axis=-1)
        x_g = self.dropout_1(x_g)

        # Cell type.
        c_e = self.encoder_cell_type_2(c)
        x_c = self.decoder_cell_type_2(c_e)
        x_c = self.dropout_2(x_c)

        # Skip-connection autoencoder and final layers.
        x = tf.concat([t, g_e, c_e], axis=-1)
        for i in range(self.nn_arch['num_sc_ae']):
            x = self.sc_aes[i](x)
            x = self.dropout_3(x)

        # Final layers.
        x = self.dense_4_1(x)
        x = self.dense_4_2(x)
        x = self.dense_4_3(x)
        x = self.dropout_4_3(x)


        # Normalize x.
        if self.conf['loss_type'] == LOSS_TYPE_MULTI_LABEL:
            x1 = self.dense_4_4(x)
        elif self.conf['loss_type'] == LOSS_TYPE_ADDITIVE_ANGULAR_MARGIN:
            x = x / tf.sqrt(tf.reduce_sum(tf.square(x), axis=1, keepdims=True))
            x1 = self.dense_4_4(x)
        else:
            raise ValueError('loss type is not valid.')
        outputs = [x_g, x_c, x1]
        return outputs

    def get_config(self):
        """Get configuration."""
        config = {'conf': self.conf}
        base_config = super(_MoAPredictor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MoAPredictor(object):
    """MoA predictor."""

    # Constants.
    MODEL_PATH = 'MoA_predictor'
    OUTPUT_FILE_NAME = 'submission.csv'
    EVALUATION_FILE_NAME = 'eval.csv'

    def __init__(self, conf):
        """
        Parameters
        ----------
        conf: Dictionary
            Configuration dictionary.
        """
        # Initialize.
        self.conf = conf
        self.raw_data_path = self.conf['raw_data_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']
        self.model_loading = self.conf['model_loading']

        # Create weight for classification imbalance.
        W = self._create_W()

        # with strategy.scope():
        if self.conf['cv_type'] == CV_TYPE_TRAIN_VAL_SPLIT:
            if self.model_loading:
                self.model = load_model(self.MODEL_PATH + '.h5'
                                        , custom_objects={'MoALoss': MoALoss
                                            , 'MoAMetric': MoAMetric
                                            , '_MoAPredictor': _MoAPredictor}
                                        , compile=False)
                #self.model = load_model(self.MODEL_PATH, compile=False)
                opt = optimizers.Adam(lr=self.hps['lr']
                                      , beta_1=self.hps['beta_1']
                                      , beta_2=self.hps['beta_2']
                                      , decay=self.hps['decay'])
                self.model.compile(optimizer=opt
                                   , loss=['mse', 'mse', MoALoss(W
                                                                  , self.nn_arch['additive_margin']
                                                                  , self.hps['ls']
                                                                  , self.nn_arch['scale']
                                                                  , loss_type=self.conf['loss_type'])]
                              , loss_weights=self.hps['loss_weights']
                              , metrics=[['mse'], ['mse'], [MoAMetric(self.hps['sn_t'])]]
                              , run_eagerly=False)
            else:
                # Design the MoA prediction model.
                # Input.
                input_t = Input(shape=(self.nn_arch['d_treatment_type'],))
                input_g = Input(shape=(self.nn_arch['d_gene_exp'],))
                input_c = Input(shape=(self.nn_arch['d_cell_type'],))

                outputs = _MoAPredictor(self.conf, name='moap')([input_t, input_g, input_c])

                opt = optimizers.Adam(lr=self.hps['lr']
                                      , beta_1=self.hps['beta_1']
                                      , beta_2=self.hps['beta_2']
                                      , decay=self.hps['decay'])

                self.model = Model(inputs=[input_t, input_g, input_c], outputs=outputs)
                self.model.compile(optimizer=opt
                                   , loss=['mse', 'mse', MoALoss(W
                                                                  , self.nn_arch['additive_margin']
                                                                  , self.hps['ls']
                                                                  , self.nn_arch['scale']
                                                                  , loss_type=self.conf['loss_type'])]
                              , loss_weights=self.hps['loss_weights']
                              , metrics=[['mse'], ['mse'], [MoAMetric(self.hps['sn_t'])]]
                              , run_eagerly=False)
                self.model.summary()
        elif self.conf['cv_type'] == CV_TYPE_K_FOLD:
            self.k_fold_models = []

            if self.model_loading:
                opt = optimizers.Adam(lr=self.hps['lr']
                                      , beta_1=self.hps['beta_1']
                                      , beta_2=self.hps['beta_2']
                                      , decay=self.hps['decay'])

                # load models for K-fold.
                for i in range(self.nn_arch['k_fold']):
                    self.k_fold_models.append(load_model(self.MODEL_PATH + '_' + str(i) + '.h5'
                                                        , custom_objects={'MoALoss': MoALoss
                                                            , 'MoAMetric': MoAMetric
                                                            , '_MoAPredictor': _MoAPredictor}
                                                         , compile=False))
                    self.k_fold_models[i].compile(optimizer=opt
                                   , loss=['mse', 'mse', MoALoss(W
                                                                  , self.nn_arch['additive_margin']
                                                                  , self.hps['ls']
                                                                  , self.nn_arch['scale']
                                                                  , loss_type=self.conf['loss_type'])]
                              , loss_weights=self.hps['loss_weights']
                              , metrics=[['mse'], ['mse'], [MoAMetric(self.hps['sn_t'])]]
                              , run_eagerly=False)
            else:
                # Create models for K-fold.
                for i in range(self.nn_arch['k_fold']):
                    # Design the MoA prediction model.
                    # Input.
                    input_t = Input(shape=(self.nn_arch['d_treatment_type'],))
                    input_g = Input(shape=(self.nn_arch['d_gene_exp'],))
                    input_c = Input(shape=(self.nn_arch['d_cell_type'],))

                    outputs = _MoAPredictor(self.conf, name='moap')([input_t, input_g, input_c])

                    opt = optimizers.Adam(lr=self.hps['lr']
                                          , beta_1=self.hps['beta_1']
                                          , beta_2=self.hps['beta_2']
                                          , decay=self.hps['decay'])

                    model = Model(inputs=[input_t, input_g, input_c], outputs=outputs)
                    model.compile(optimizer=opt
                                   , loss=['mse', 'mse', MoALoss(W
                                                                  , self.nn_arch['additive_margin']
                                                                  , self.hps['ls']
                                                                  , self.nn_arch['scale']
                                                                  , loss_type=self.conf['loss_type'])]
                              , loss_weights=self.hps['loss_weights']
                              , metrics=[['mse'], ['mse'], [MoAMetric(self.hps['sn_t'])]]
                              , run_eagerly=False)
                    model.summary()

                    self.k_fold_models.append(model)
        else:
            raise ValueError('cv_type is not valid.')

        # Create dataset.
        self._create_dataset()

    def _create_dataset(self):
        input_df = pd.read_csv(os.path.join(self.raw_data_path, 'train_features.csv')) #.iloc[:1024]
        input_df.cp_type = input_df.cp_type.astype('category')
        input_df.cp_type = input_df.cp_type.cat.rename_categories(range(len(input_df.cp_type.cat.categories)))
        input_df.cp_time = input_df.cp_time.astype('category')
        input_df.cp_time = input_df.cp_time.cat.rename_categories(range(len(input_df.cp_time.cat.categories)))
        input_df.cp_dose = input_df.cp_dose.astype('category')
        input_df.cp_dose = input_df.cp_dose.cat.rename_categories(range(len(input_df.cp_dose.cat.categories)))

        # Remove samples of ctl_vehicle.
        valid_indexes = input_df.cp_type == 1
        input_df = input_df[valid_indexes]
        input_df = input_df.reset_index(drop=True)

        target_scored_df = pd.read_csv(os.path.join(self.raw_data_path, 'train_targets_scored.csv')) #.iloc[:1024]
        target_scored_df = target_scored_df[valid_indexes]
        target_scored_df = target_scored_df.reset_index(drop=True)
        del target_scored_df['sig_id']
        target_scored_df.columns = range(len(target_scored_df.columns))
        n_target_samples = target_scored_df.sum().values

        if self.conf['data_aug']:
            genes = [col for col in input_df.columns if col.startswith("g-")]
            cells = [col for col in input_df.columns if col.startswith("c-")]

            features = genes + cells
            targets = [col for col in target_scored_df if col != 'sig_id']

            aug_trains = []
            aug_targets = []
            for t in [0, 1, 2]:
                for d in [0, 1]:
                    for _ in range(3):
                        train1 = input_df.loc[(input_df['cp_time'] == t) & (input_df['cp_dose'] == d)]
                        target1 = target_scored_df.loc[(input_df['cp_time'] == t) & (input_df['cp_dose'] == d)]
                        ctl1 = input_df.loc[(input_df['cp_time'] == t) & (input_df['cp_dose'] == d)].sample(
                            train1.shape[0], replace=True)
                        ctl2 = input_df.loc[(input_df['cp_time'] == t) & (input_df['cp_dose'] == d)].sample(
                            train1.shape[0], replace=True)
                        train1[genes + cells] = train1[genes + cells].values + ctl1[genes + cells].values - ctl2[
                            genes + cells].values
                        aug_trains.append(train1)
                        aug_targets.append(target1)

            input_df = pd.concat(aug_trains).reset_index(drop=True)
            target_scored_df = pd.concat(aug_targets).reset_index(drop=True)

        g_feature_names = ['g-' + str(v) for v in range(self.nn_arch['d_gene_exp'])]
        c_feature_names = ['c-' + str(v) for v in range(self.nn_arch['d_cell_type'])]
        moa_names = [v for v in range(self.nn_arch['num_moa_annotation'])]

        def get_series_from_input(idxes):
            idxes = idxes.numpy() #?
            series = input_df.iloc[idxes]

            # Treatment.
            if isinstance(idxes, np.int32) != True:
                cp_time = series['cp_time'].values.to_numpy()
                cp_dose = series['cp_dose'].values.to_numpy()
            else:
                cp_time = np.asarray(series['cp_time'])
                cp_dose = np.asarray(series['cp_dose'])

            treatment_type = cp_time * 2 + cp_dose

            # Gene expression.
            gene_exps = series[g_feature_names].values

            # Cell viability.
            cell_vs = series[c_feature_names].values

            return treatment_type, gene_exps, cell_vs


        def make_input_target_features(idxes):
            treatment_type, gene_exps, cell_vs = tf.py_function(get_series_from_input, inp=[idxes], Tout=[tf.int64, tf.float64, tf.float64])
            MoA_values = tf.py_function(get_series_from_target, inp=[idxes], Tout=tf.int32)
            return ((treatment_type, gene_exps, cell_vs), (gene_exps, cell_vs, MoA_values))


        def make_input_features(idx):
            treatment_type, gene_exps, cell_vs = tf.py_function(get_series_from_input, inp=[idx], Tout=[tf.int64, tf.float64, tf.float64])
            return treatment_type, gene_exps, cell_vs


        def make_a_target_features(idx):
            treatment_type, gene_exps, cell_vs = tf.py_function(get_series_from_input, inp=[idx], Tout=[tf.int64, tf.float64, tf.float64])
            return gene_exps, cell_vs


        def get_series_from_target(idxes):
            idxes = idxes.numpy()
            series = target_scored_df.iloc[idxes]

            # MoA annotations' values.
            MoA_values = series[moa_names].values

            return MoA_values


        def make_target_features(idx):
            MoA_values = tf.py_function(get_series_from_target, inp=[idx], Tout=tf.int32)
            return MoA_values


        def divide_inputs(input1, input2):
            return input1[0], input1[1], input2


        if self.conf['cv_type'] == CV_TYPE_TRAIN_VAL_SPLIT:
            if self.conf['dataset_type'] == DATASET_TYPE_PLAIN:
                train_val_index = np.arange(len(input_df))
                #np.random.shuffle(train_val_index)
                num_val = int(self.conf['val_ratio'] * len(input_df))
                num_tr = len(input_df) - num_val
                train_index = train_val_index[:num_tr]
                val_index = train_val_index[num_tr:]
                self.train_index = train_index
                self.val_index = val_index

                # Training dataset.
                input_dataset = tf.data.Dataset.from_tensor_slices(train_index)
                input_dataset = input_dataset.map(make_input_features)

                a_target_dataset = tf.data.Dataset.from_tensor_slices(train_index)
                a_target_dataset = a_target_dataset.map(make_a_target_features)

                target_dataset = tf.data.Dataset.from_tensor_slices(train_index)
                target_dataset = target_dataset.map(make_target_features)

                f_target_dataset = tf.data.Dataset.zip((a_target_dataset, target_dataset)).map(divide_inputs)

                # Inputs and targets.
                tr_dataset = tf.data.Dataset.zip((input_dataset, f_target_dataset))
                tr_dataset = tr_dataset.shuffle(buffer_size=self.hps['batch_size'] * 5
                                                , reshuffle_each_iteration=True).repeat().batch(self.hps['batch_size'])
                self.step = len(train_index) // self.hps['batch_size']

                # Validation dataset.
                input_dataset = tf.data.Dataset.from_tensor_slices(val_index)
                input_dataset = input_dataset.map(make_input_features)

                a_target_dataset = tf.data.Dataset.from_tensor_slices(val_index)
                a_target_dataset = a_target_dataset.map(make_a_target_features)

                target_dataset = tf.data.Dataset.from_tensor_slices(val_index)
                target_dataset = target_dataset.map(make_target_features)

                f_target_dataset = tf.data.Dataset.zip((a_target_dataset, target_dataset)).map(divide_inputs)

                # Inputs and targets.
                val_dataset = tf.data.Dataset.zip((input_dataset, f_target_dataset))
                val_dataset = val_dataset.batch(self.hps['batch_size'])

                self.trval_dataset = (tr_dataset, val_dataset)
            elif self.conf['dataset_type'] == DATASET_TYPE_BALANCED:
                MoA_p_sets = []
                val_index = []
                for col in target_scored_df.columns:
                    s = target_scored_df.iloc[:, col]
                    s = s[s == 1]
                    s = list(s.index)
                    #shuffle(s)
                    n_val = int(n_target_samples[col] * self.conf['val_ratio'])

                    if n_val != 0:
                        tr_set = s[:int(-1.0 * n_val)]
                        val_set = s[int(-1.0 * n_val):]
                        MoA_p_sets.append(tr_set)
                        val_index += val_set
                    else:
                        MoA_p_sets.append(s)

                df = target_scored_df.sum(axis=1)
                df = df[df == 0]
                no_MoA_p_set = list(df.index)
                #shuffle(no_MoA_p_set)
                val_index += no_MoA_p_set[int(-1.0 * len(no_MoA_p_set) * self.conf['val_ratio']):]

                MoA_p_sets.append(no_MoA_p_set[:int(-1.0 * len(no_MoA_p_set) * self.conf['val_ratio'])])

                idxes = []
                for i in range(self.hps['rep']):
                    for col in range(len(target_scored_df.columns) + 1):
                        if len(MoA_p_sets[col]) >= (i + 1):
                            idx = MoA_p_sets[col][i]
                        else:
                            idx = np.random.choice(MoA_p_sets[col], size=1, replace=True)[0]
                        idxes.append(idx)

                train_index = idxes
                self.train_index = train_index
                self.val_index = val_index

                # Training dataset.
                tr_dataset = tf.data.Dataset.from_tensor_slices(train_index)

                # Inputs and targets.
                tr_dataset = tr_dataset.shuffle(buffer_size=self.hps['batch_size'] * 5
                                                , reshuffle_each_iteration=True).repeat().batch(self.hps['batch_size']).map(make_input_target_features
                                                , num_parallel_calls=tf.data.experimental.AUTOTUNE)
                self.step = len(train_index) // self.hps['batch_size']

                # Validation dataset.
                val_dataset = tf.data.Dataset.from_tensor_slices(val_index)

                # Inputs and targets.
                val_dataset = val_dataset.batch(self.hps['batch_size']).map(make_input_target_features
                                                , num_parallel_calls=tf.data.experimental.AUTOTUNE)

                # Save datasets.
                #tf.data.experimental.save(tr_dataset, './tr_dataset')
                #tf.data.experimental.save(val_dataset, './val_dataset')

                self.trval_dataset = (tr_dataset, val_dataset)
            else:
                raise ValueError('dataset type is not valid.')
        elif self.conf['cv_type'] == CV_TYPE_K_FOLD:
            stratified_kfold = StratifiedKFold(n_splits=self.nn_arch['k_fold'])
            # group_kfold = GroupKFold(n_splits=self.nn_arch['k_fold'])
            self.k_fold_trval_datasets = []

            for train_index, val_index in stratified_kfold.split(input_df, input_df.cp_type):
                # Training dataset.
                input_dataset = tf.data.Dataset.from_tensor_slices(train_index)
                input_dataset = input_dataset.map(make_input_features)

                a_target_dataset = tf.data.Dataset.from_tensor_slices(train_index)
                a_target_dataset = a_target_dataset.map(make_a_target_features)

                target_dataset = tf.data.Dataset.from_tensor_slices(train_index)
                target_dataset = target_dataset.map(make_target_features)

                f_target_dataset = tf.data.Dataset.zip((a_target_dataset, target_dataset)).map(divide_inputs)

                # Inputs and targets.
                tr_dataset = tf.data.Dataset.zip((input_dataset, f_target_dataset))
                tr_dataset = tr_dataset.shuffle(buffer_size=self.hps['batch_size'] * 5
                                                , reshuffle_each_iteration=True).repeat().batch(self.hps['batch_size'])
                self.step = len(train_index) // self.hps['batch_size']

                # Validation dataset.
                input_dataset = tf.data.Dataset.from_tensor_slices(val_index)
                input_dataset = input_dataset.map(make_input_features)

                a_target_dataset = tf.data.Dataset.from_tensor_slices(val_index)
                a_target_dataset = a_target_dataset.map(make_a_target_features)

                target_dataset = tf.data.Dataset.from_tensor_slices(val_index)
                target_dataset = target_dataset.map(make_target_features)

                f_target_dataset = tf.data.Dataset.zip((a_target_dataset, target_dataset)).map(divide_inputs)

                # Inputs and targets.
                val_dataset = tf.data.Dataset.zip((input_dataset, f_target_dataset))
                val_dataset = val_dataset.batch(self.hps['batch_size'])

                self.k_fold_trval_datasets.append((tr_dataset, val_dataset))
        else:
            raise ValueError('cv_type is not valid.')

    def _create_W(self):
        target_scored_df = pd.read_csv(os.path.join(self.raw_data_path, 'train_targets_scored.csv'))
        del target_scored_df['sig_id']

        weights = []
        for c in target_scored_df.columns:
            s = target_scored_df[c]
            s = s.value_counts()
            s = s / s.sum()
            weights.append(s.values)

        weight = np.expand_dims(np.array(weights), axis=0)

        return weight

    def train(self):
        """Train."""
        reduce_lr = ReduceLROnPlateau(monitor='val_loss'
                                      , factor=self.hps['reduce_lr_factor']
                                      , patience=3
                                      , min_lr=1.e-8
                                      , verbose=1)
        tensorboard = TensorBoard(histogram_freq=1
                                  , write_graph=True
                                  , write_images=True
                                  , update_freq='epoch')

        earlystopping = EarlyStopping(monitor='val_loss'
                                      , min_delta=0
                                      , patience=5
                                      , verbose=1
                                      , mode='auto')

        '''
        def schedule_lr(e_i):
            self.hps['lr'] = self.hps['reduce_lr_factor'] * self.hps['lr']
            return self.hps['lr']

        lr_scheduler = LearningRateScheduler(schedule_lr, verbose=1)
        '''

        if self.conf['cv_type'] == CV_TYPE_TRAIN_VAL_SPLIT:
            model_check_point = ModelCheckpoint(self.MODEL_PATH + '.h5'
                                                , monitor='val_loss'
                                                , verbose=1
                                                , save_best_only=True)

            hist = self.model.fit(self.trval_dataset[0]
                                                , steps_per_epoch=self.step
                                                , epochs=self.hps['epochs']
                                                , verbose=1
                                                , max_queue_size=80
                                                , workers=4
                                                , use_multiprocessing=False
                                                , callbacks=[model_check_point, earlystopping] #, reduce_lr] #, tensorboard]
                                                , validation_data=self.trval_dataset[1]
                                                , validation_freq=1
                                                , shuffle=True)
        elif self.conf['cv_type'] == CV_TYPE_K_FOLD:
            for i in range(self.nn_arch['k_fold']):
                model_check_point = ModelCheckpoint(self.MODEL_PATH + '_' + str(i) + '.h5'
                                                    , monitor='loss'
                                                    , verbose=1
                                                    , save_best_only=True)

                hist = self.k_fold_models[i].fit(self.k_fold_trval_datasets[i][0]
                                                    , steps_per_epoch=self.step
                                                    , epochs=self.hps['epochs']
                                                    , verbose=1
                                                    , max_queue_size=80
                                                    , workers=4
                                                    , use_multiprocessing=False
                                                    , callbacks=[model_check_point, earlystopping] #reduce_lr] #, tensorboard]
                                                    , validation_data=self.k_fold_trval_datasets[i][1]
                                                    , validation_freq=1
                                                    , shuffle=True)
        else:
            raise ValueError('cv_type is not valid.')

        #print('Save the model.')
        #self.model.save(self.MODEL_PATH, save_format='h5')
        # self.model.save(self.MODEL_PATH, save_format='tf')
        return hist

    def evaluate(self):
        """Evaluate."""
        assert self.conf['cv_type'] == CV_TYPE_TRAIN_VAL_SPLIT

        input_df = pd.read_csv(os.path.join(self.raw_data_path, 'train_features.csv')) #.iloc[:1024]
        input_df.cp_type = input_df.cp_type.astype('category')
        input_df.cp_type = input_df.cp_type.cat.rename_categories(range(len(input_df.cp_type.cat.categories)))
        input_df.cp_time = input_df.cp_time.astype('category')
        input_df.cp_time = input_df.cp_time.cat.rename_categories(range(len(input_df.cp_time.cat.categories)))
        input_df.cp_dose = input_df.cp_dose.astype('category')
        input_df.cp_dose = input_df.cp_dose.cat.rename_categories(range(len(input_df.cp_dose.cat.categories)))

        # Remove samples of ctl_vehicle.
        valid_indexes = input_df.cp_type == 1  # ?
        target_scored_df = pd.read_csv(os.path.join(self.raw_data_path, 'train_targets_scored.csv')) #.iloc[:1024]
        target_scored_df = target_scored_df.loc[self.val_index]
        MoA_annots = target_scored_df.columns[1:]

        def make_input_features(inputs):
            # Treatment.
            cp_time = inputs['cp_time']
            cp_dose = inputs['cp_dose']

            treatment_type = cp_time * 2 + cp_dose

            # Gene expression.
            gene_exps = [inputs['g-' + str(v)] for v in range(self.nn_arch['d_gene_exp'])]
            gene_exps = tf.stack(gene_exps, axis=0)

            # Cell viability.
            cell_vs = [inputs['c-' + str(v)] for v in range(self.nn_arch['d_cell_type'])]
            cell_vs = tf.stack(cell_vs, axis=0)

            return (tf.expand_dims(treatment_type, axis=-1), gene_exps, cell_vs)

        # Validation dataset.
        val_dataset = tf.data.Dataset.from_tensor_slices(input_df.loc[self.val_index].to_dict('list'))
        val_dataset = val_dataset.map(make_input_features)

        val_iter = val_dataset.as_numpy_iterator()

        # Predict MoAs.
        sig_id_list = []
        MoAs = [[] for _ in range(len(MoA_annots))]

        for i, d in tqdm(enumerate(val_iter)):
            t, g, c = d
            id = target_scored_df['sig_id'].iloc[i]
            t = np.expand_dims(t, axis=0)
            g = np.expand_dims(g, axis=0)
            c = np.expand_dims(c, axis=0)

            if self.conf['cv_type'] == CV_TYPE_TRAIN_VAL_SPLIT:
                _, _, result = self.model.layers[-1]([t, g, c])  # self.model.predict([t, g, c])
                result = np.squeeze(result, axis=0)
                #result = np.exp(result) / (np.sum(np.exp(result), axis=0) + epsilon)

                for i, MoA in enumerate(result):
                    MoAs[i].append(MoA)
            elif self.conf['cv_type'] == CV_TYPE_K_FOLD:
                # Conduct ensemble prediction.
                result_list = []

                for i in range(self.nn_arch['k_fold']):
                    _, _, result = self.k_fold_models[i].predict([t, g, c])
                    result = np.squeeze(result, axis=0)
                    #result = np.exp(result) / (np.sum(np.exp(result), axis=0) + epsilon)
                    result_list.append(result)

                result_mean = np.asarray(result_list).mean(axis=0)

                for i, MoA in enumerate(result_mean):
                    MoAs[i].append(MoA)
            else:
                raise ValueError('cv_type is not valid.')

            sig_id_list.append(id)

        # Save the result.
        result_dict = {'sig_id': sig_id_list}
        for i, MoA_annot in enumerate(MoA_annots):
            result_dict[MoA_annot] = MoAs[i]

        submission_df = pd.DataFrame(result_dict)
        submission_df.to_csv(self.OUTPUT_FILE_NAME, index=False)

        target_scored_df.to_csv('gt.csv', index=False)

    def test(self):
        """Test."""

        # Create the test dataset.
        input_df = pd.read_csv(os.path.join(self.raw_data_path, 'test_features.csv'))
        input_df.cp_type = input_df.cp_type.astype('category')
        input_df.cp_type = input_df.cp_type.cat.rename_categories(range(len(input_df.cp_type.cat.categories)))
        input_df.cp_time = input_df.cp_time.astype('category')
        input_df.cp_time = input_df.cp_time.cat.rename_categories(range(len(input_df.cp_time.cat.categories)))
        input_df.cp_dose = input_df.cp_dose.astype('category')
        input_df.cp_dose = input_df.cp_dose.cat.rename_categories(range(len(input_df.cp_dose.cat.categories)))

        # Remove samples of ctl_vehicle.
        valid_indexes = input_df.cp_type == 1 #?
        target_scored_df = pd.read_csv(os.path.join(self.raw_data_path, 'train_targets_scored.csv'))
        MoA_annots = target_scored_df.columns[1:]

        def make_input_features(inputs):
            id_ = inputs['sig_id']
            cp_type = inputs['cp_type']

            # Treatment.
            cp_time = inputs['cp_time']
            cp_dose = inputs['cp_dose']

            treatment_type = cp_time * 2 + cp_dose

            # Gene expression.
            gene_exps = [inputs['g-' + str(v)] for v in range(self.nn_arch['d_gene_exp'])]
            gene_exps = tf.stack(gene_exps, axis=0)

            # Cell viability.
            cell_vs = [inputs['c-' + str(v)] for v in range(self.nn_arch['d_cell_type'])]
            cell_vs = tf.stack(cell_vs, axis=0)

            return (id_, cp_type, tf.expand_dims(treatment_type, axis=-1), gene_exps, cell_vs)

        test_dataset = tf.data.Dataset.from_tensor_slices(input_df.to_dict('list'))
        test_dataset = test_dataset.map(make_input_features)
        test_iter = test_dataset.as_numpy_iterator()

        # Predict MoAs.
        sig_id_list = []
        MoAs = [[] for _ in range(len(MoA_annots))]

        def cal_prob(logit):
            a = logit
            a = (a + 1.0) / 2.0
            a = tf.where(tf.math.greater(a, self.hps['sn_t']), a, 0.0)
            a = self.hps['m1'] * a + self.hps['m2']
            p_h = tf.sigmoid(a).numpy()
            return p_h

        def cal_prob_2(logit):
            y_pred = logit
            E = tf.reduce_mean(tf.math.exp(y_pred), axis=-1, keepdims=True)
            E_2 = tf.reduce_mean(tf.square(tf.math.exp(y_pred)), axis=-1, keepdims=True)
            S = tf.sqrt(E_2 - tf.square(E))

            e_A = (tf.exp(y_pred) - E) / (S + epsilon)
            e_A_p = tf.where(tf.math.greater(e_A, self.hps['sn_t']), self.hps['sn_t'], 0.0)
            p_h = e_A_p / (tf.reduce_sum(e_A_p, axis=-1, keepdims=True) + epsilon)
            return p_h.numpy()

        def cal_prob_3(logit):
            A = logit
            A = (A + 1.0) / 2.0

            E = tf.reduce_mean(A, axis=-1, keepdims=True)
            E_2 = tf.reduce_mean(tf.square(A), axis=-1, keepdims=True)
            S = tf.sqrt(E_2 - tf.square(E))

            #S_N = tf.abs(A - E) / (S + epsilon)
            S_N = (A - E) / (S + epsilon)
            #S_N = tf.where(tf.math.greater(S_N, self.hps['sn_t']), S_N, 0.0)
            A_p = self.hps['m1'] * S_N + self.hps['m2']
            #P_h = tf.clip_by_value(A_p / 10.0, clip_value_min=0.0, clip_value_max=1.0)
            P_h = tf.sigmoid(A_p)
            return P_h.numpy()

        def cal_prob_4(logit):
            a = logit
            p_h = tf.sigmoid(a).numpy()
            return p_h

        for id_, cp_type, t, g, c in tqdm(test_iter):
            id_ = id_.decode('utf8') #?
            t = np.expand_dims(t, axis=0)
            g = np.expand_dims(g, axis=0)
            c = np.expand_dims(c, axis=0)

            if self.conf['cv_type'] == CV_TYPE_TRAIN_VAL_SPLIT:
                #_, _, result = self.model.layers[-1]([t, g, c]) #self.model.predict([t, g, c])
                _, _, result = self.model.predict([t, g, c])
                result = np.squeeze(result, axis=0)

                if cp_type == 1:
                    if self.conf['loss_type'] == LOSS_TYPE_MULTI_LABEL:
                        result = cal_prob_4(result)
                    elif self.conf['loss_type'] == LOSS_TYPE_ADDITIVE_ANGULAR_MARGIN:
                        result = cal_prob_3(result)
                    else:
                        raise ValueError('loss type is not valid.')
                else:
                    result = np.zeros((len(result)))

                for i, MoA in enumerate(result):
                    MoAs[i].append(MoA)
            elif self.conf['cv_type'] == CV_TYPE_K_FOLD:
                # Conduct ensemble prediction.
                result_list = []

                for i in range(self.nn_arch['k_fold']):
                    _, _, result = self.k_fold_models[i].predict([t, g, c])
                    result = np.squeeze(result, axis=0)

                    if cp_type == 1:
                        if self.conf['loss_type'] == LOSS_TYPE_MULTI_LABEL:
                            result = cal_prob_4(result)
                        elif self.conf['loss_type'] == LOSS_TYPE_ADDITIVE_ANGULAR_MARGIN:
                            result = cal_prob_3(result)
                        else:
                            raise ValueError('loss type is not valid.')
                    else:
                        result = np.zeros((len(result)))

                    result_list.append(result)

                result_mean = np.asarray(result_list).mean(axis=0)

                for i, MoA in enumerate(result_mean):
                    MoAs[i].append(MoA)
            else:
                raise ValueError('cv_type is not valid.')

            sig_id_list.append(id_)

        # Save the result.
        result_dict = {'sig_id': sig_id_list}
        for i, MoA_annot in enumerate(MoA_annots):
            result_dict[MoA_annot] = MoAs[i]

        submission_df = pd.DataFrame(result_dict)
        submission_df.to_csv(self.OUTPUT_FILE_NAME, index=False)


def main():
    """Main."""
    seed = int(time.time())
    # seed = 1606208227
    print(f'Seed:{seed}')
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load configuration.
    with open("MoA_pred_conf.json", 'r') as f:
        conf = json.load(f)

    if conf['mode'] == 'train':
        # Train.
        model = MoAPredictor(conf)

        ts = time.time()
        model.train()
        te = time.time()

        print('Elasped time: {0:f}s'.format(te - ts))
    elif conf['mode'] == 'evaluate':
        # Evaluate.
        model = MoAPredictor(conf)

        ts = time.time()
        model.evaluate()
        te = time.time()

        print('Elasped time: {0:f}s'.format(te - ts))
    elif conf['mode'] == 'test':
        model = MoAPredictor(conf)

        ts = time.time()
        model.test()
        te = time.time()

        print('Elasped time: {0:f}s'.format(te - ts))
    else:
        raise ValueError('Mode is not valid.')


if __name__ == '__main__':
    main()