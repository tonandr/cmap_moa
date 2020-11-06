'''
Created on Oct 25, 2020
@author: Inwoo Chung (gutomitai@gmail.com)
'''

import os
import time
import json
import copy
from itertools import combinations

from scipy.special import comb
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, Dense, Concatenate, Dropout
from tensorflow.keras.layers import LSTM, Bidirectional, BatchNormalization, LayerNormalization
from tensorflow.python.keras.layers import Embedding, Layer
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint

from ku.composite_layer import Transformer\
    , SIMILARITY_TYPE_DIFF_ABS\
    , SIMILARITY_TYPE_PLAIN\
    , SIMILARITY_TYPE_SCALED\
    , SIMILARITY_TYPE_GENERAL\
    , SIMILARITY_TYPE_ADDITIVE

from ku.composite_layer import DenseBatchNormalization
from ku.backprop import make_decoder_from_encoder\
    , make_autoencoder_from_encoder\
    , make_autoencoder_with_sym_sc
from ku.gnn_layer import GraphConvolutionNetwork

# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# Constants.
DEBUG = True

MODE_TRAIN = 0
MODE_VAL = 1

MODEL_TYPE_GAE = 'gae'
MODEL_TYPE_EXTRACTOR = 'extractor'

CV_TYPE_TRAIN_VAL_SPLIT = 'train_val_split'
CV_TYPE_K_FOLD = 'k_fold'

epsilon = 1e-7

A = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])


def MoA_loss(y_true, y_pred):
    loss = tf.reduce_mean(y_pred)
    return loss


def MoA_metric(y_true, y_pred):
    y_pred = tf.maximum(tf.minimum(y_pred, 1.0 - 1e-15), 1e-15)
    y_true = tf.cast(y_true, dtype=tf.float32)

    log_loss = -1.0 * (y_true * tf.math.log(y_pred + epsilon) + (1.0 - y_true) * tf.math.log(1.0 - y_pred + epsilon))
    log_loss_mean = tf.reduce_mean(log_loss, axis=0) #?
    loss = tf.reduce_mean(log_loss_mean, axis=0)
    return loss

MoA_metric.__name__ = 'MoA_metric'


class CMAPFeatureGAE(Layer):
    def __init__(self, conf, **kwargs):
        super(CMAPFeatureGAE, self).__init__(**kwargs)

        # Initialize.
        self.conf = conf
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']
        gcn_d_outs = self.nn_arch['gcn_d_outs']
        gcn_d_outs_r = copy.copy(gcn_d_outs)
        gcn_d_outs_r.reverse()

        # Design layers.
        # First layers.
        self.embed_treatment_type_0 = Embedding(self.nn_arch['num_treatment_type']
                                           , self.nn_arch['d_input_feature'])
        self.dense_treatment_type_0 = Dense(self.nn_arch['d_input_feature']
                                       , activation='relu')
        #self.batch_normalization_0_1 = BatchNormalization()
        self.dense_gene_exp_0 = Dense(self.nn_arch['d_input_feature']
                                 , activation='relu')
        #self.batch_normalization_0_2 = BatchNormalization()
        self.dense_cell_type_0 = Dense(self.nn_arch['d_input_feature']
                                  , activation='relu')
        #self.batch_normalization_0_3 = BatchNormalization()

        self.layer_normalization_0_1 = LayerNormalization()
        self.layer_normalization_0_2 = LayerNormalization()
        self.layer_normalization_0_3 = LayerNormalization()

        self.concat_0 = Concatenate(axis=1)

        # GCN encoder.
        input_feature_1 = Input(shape=(self.nn_arch['n_node'], self.nn_arch['d_input_feature'])
                                 , dtype='float32', name='input_feature_1')
        input_A_1 = Input(shape=(self.nn_arch['n_node'], self.nn_arch['n_node'])
                        , dtype='float32', name='input_A_1')

        x_1 = input_feature_1
        A_1 = input_A_1
        for i, d_out in enumerate(gcn_d_outs):
            x_1, A_1 = GraphConvolutionNetwork(self.nn_arch['n_node']
                                               , d_out
                                               , output_adjacency=True
                                               , activation=keras.activations.relu)([x_1, A_1])

        output_1 = [x_1, A_1]

        self.gcn_encoder_1 = Model(inputs=[input_feature_1, input_A_1], outputs=output_1)

        # GCN decoder.
        input_feature_2 = Input(shape=(self.nn_arch['n_node'], gcn_d_outs_r[0])
                                 , dtype='float32', name='input_feature_2')
        input_A_2 = Input(shape=(self.nn_arch['n_node'], self.nn_arch['n_node'])
                        , dtype='float32', name='input_A_2')

        gcn_d_outs_r = gcn_d_outs_r[1:] + [self.nn_arch['d_input_feature']]

        x_2 = input_feature_2
        A_2 = input_A_2
        for i, d_out in enumerate(gcn_d_outs_r):
            x_2, A_2 = GraphConvolutionNetwork(self.nn_arch['n_node']
                                               , d_out
                                               , output_adjacency=True
                                               , activation='swish')([x_2, A_2])

        output_2 = [x_2, A_2]

        self.gcn_decoder_1 = Model(inputs=[input_feature_2, input_A_2], outputs=output_2)

    def call(self, inputs):
        t = inputs[0]
        g = inputs[1]
        c = inputs[2]
        A_ = inputs[3]

        # First layer.
        # X.
        t = self.embed_treatment_type_0(t)
        t = tf.reshape(t, (-1, self.nn_arch['d_input_feature']))
        t = self.dense_treatment_type_0(t)
        #t = self.batch_normalization_0_1(t)

        g = self.dense_gene_exp_0(g)
        #g = self.batch_normalization_0_2(g)

        c = self.dense_cell_type_0(c)
        #c = self.batch_normalization_0_3(c)

        t = self.layer_normalization_0_1(t)
        g = self.layer_normalization_0_2(g)
        c = self.layer_normalization_0_3(c)

        t = tf.expand_dims(t, axis=1)
        g = tf.expand_dims(g, axis=1)
        c = tf.expand_dims(c, axis=1)

        X = self.concat_0([t, g, c])

        # GCN encoding layer.
        Z, _ = self.gcn_encoder_1([X, A_])

        # GCN decoding layer for graph structure.
        #A_h = tf.sigmoid(K.batch_dot(Z, tf.transpose(Z, perm=[0, 2, 1])))
        A_h = K.batch_dot(Z, tf.transpose(Z, perm=[0, 2, 1])) #+ 1e-15

        # GCN decoding layer for node feature.
        X_h, _ = self.gcn_decoder_1([Z, A_])

        outputs = [X, X_h, A_h]
        return outputs

    def get_config(self):
        """Get configuration."""
        config = {'conf': self.conf}
        base_config = super(CMAPFeatureGAE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CMAPFeatureExtractor(Layer):
    def __init__(self, conf, A, **kwargs):
        super(CMAPFeatureExtractor, self).__init__(**kwargs)

        # Initialize.
        self.conf = conf
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']

        self.A = A

        # Design layers.
        # CMAPFeatureGAE.
        self.gae_0 = CMAPFeatureGAE(self.conf)

    def call(self, inputs):
        t_1 = inputs[0]
        g_1 = inputs[1]
        c_1 = inputs[2]
        A_1 = inputs[3]

        t_2 = inputs[4]
        g_2 = inputs[5]
        c_2 = inputs[6]
        A_2 = inputs[7]

        # First layer.
        # X_1.
        t_1 = self.gae_0.embed_treatment_type_0(t_1)
        t_1 = tf.reshape(t_1, (-1, self.nn_arch['d_input_feature']))
        t_1 = self.gae_0.dense_treatment_type_0(t_1)
        #t_1 = self.batch_normalization_0_1(t_1)

        g_1 = self.gae_0.dense_gene_exp_0(g_1)
        #g_1 = self.batch_normalization_0_2(g_1)

        c_1 = self.gae_0.dense_cell_type_0(c_1)
        #c_1 = self.batch_normalization_0_3(c_1)

        t_1 = self.gae_0.layer_normalization_0_1(t_1)
        g_1 = self.gae_0.layer_normalization_0_2(g_1)
        c_1 = self.gae_0.layer_normalization_0_3(c_1)

        t_1 = tf.expand_dims(t_1, axis=1)
        g_1 = tf.expand_dims(g_1, axis=1)
        c_1 = tf.expand_dims(c_1, axis=1)

        X_1 = self.gae_0.concat_0([t_1, g_1, c_1])

        # X_2.
        t_2 = self.gae_0.embed_treatment_type_0(t_2)
        t_2 = tf.reshape(t_2, (-1, self.nn_arch['d_input_feature']))
        t_2 = self.gae_0.dense_treatment_type_0(t_2)
        #t_2 = self.batch_normalization_0_1(t_2)

        g_2 = self.gae_0.dense_gene_exp_0(g_2)
        #g_2 = self.batch_normalization_0_2(g_2)

        c_2 = self.gae_0.dense_cell_type_0(c_2)
        #c_2 = self.batch_normalization_0_3(c_2)

        t_2 = self.gae_0.layer_normalization_0_1(t_2)
        g_2 = self.gae_0.layer_normalization_0_2(g_2)
        c_2 = self.gae_0.layer_normalization_0_3(c_2)

        t_2 = tf.expand_dims(t_2, axis=1)
        g_2 = tf.expand_dims(g_2, axis=1)
        c_2 = tf.expand_dims(c_2, axis=1)

        X_2 = self.gae_0.concat_0([t_2, g_2, c_2])

        # Get pair latent features.
        Z_1, _ = self.gae_0.gcn_encoder_1([X_1, A_1])
        Z_2, _ = self.gae_0.gcn_encoder_1([X_2, A_2])

        # Loss for intra-class.
        Z_loss = tf.square(Z_1 - Z_2) # Valid distance metric?

        # Get pair A_hs and X_hs. Normalization?
        t_1 = inputs[0]
        g_1 = inputs[1]
        c_1 = inputs[2]
        A_1 = inputs[3]

        t_2 = inputs[4]
        g_2 = inputs[5]
        c_2 = inputs[6]
        A_2 = inputs[7]

        X_1, X_h_1, A_h_1 = self.gae_0([t_1, g_1, c_1, A_1])
        X_2, X_h_2, A_h_2 = self.gae_0([t_2, g_2, c_2, A_2])

        X_1_loss = tf.square(X_1 - X_h_1)
        X_2_loss = tf.square(X_2 - X_h_2)
        X_loss = (X_1_loss + X_2_loss) / 2.0

        #A_1_loss = tf.sqrt(tf.square(tf.cast(self.A, dtype=tf.float32) - A_h_1))
        #A_2_loss = tf.sqrt(tf.square(tf.cast(self.A, dtype=tf.float32) - A_h_2))
        #A_loss = (A_1_loss + A_2_loss) / 2.0

        outputs = [Z_loss, X_loss] #, A_loss]

        return outputs

    def get_config(self):
        """Get configuration."""
        config = {'conf': self.conf
                  , 'A': self.A}
        base_config = super(CMAPFeatureExtractor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MoAPredictorGNN(object):
    """MoA predictor based on graph neural network."""

    # Constants.
    MODEL_PATH = 'MoA_predictor'
    MODEL_GAE_PATH = 'GAE'
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

        # with tpu_strategy.scope():
        if self.conf['model_type'] == MODEL_TYPE_EXTRACTOR:
            if self.conf['cv_type'] == CV_TYPE_TRAIN_VAL_SPLIT:
                if self.model_loading:
                    self.model = load_model(self.MODEL_PATH + '.h5'
                                                , custom_objects={'MoA_loss': MoA_loss
                                                , 'MoA_metric': MoA_metric
                                                , 'CMAPFeatureGAE': CMAPFeatureGAE
                                                , 'CMAPFeatureExtractor': CMAPFeatureExtractor}
                                            , compile=False)
                    opt = optimizers.Adam(lr=self.hps['lr']
                                          , beta_1=self.hps['beta_1']
                                          , beta_2=self.hps['beta_2']
                                          , decay=self.hps['decay'])
                    self.model.compile(optimizer=opt
                                  , loss=MoA_loss
                                  , loss_weights=self.hps['loss_weights']
                                  , run_eagerly=False)
                else:
                    # Design the MoA prediction model.
                    # Input.
                    input_t_1 = Input(shape=(self.nn_arch['d_treatment_type'],), dtype='float32', name='input_t_1')
                    input_g_1 = Input(shape=(self.nn_arch['d_gene_exp'],), dtype='float32', name='input_g_1')
                    input_c_1 = Input(shape=(self.nn_arch['d_cell_type'],), dtype='float32', name='input_c_1')
                    input_A_1 = Input(shape=(self.nn_arch['n_node'], self.nn_arch['n_node'])
                                      , dtype='float32', name='input_A_1')

                    input_t_2 = Input(shape=(self.nn_arch['d_treatment_type'],), dtype='float32', name='input_t_2')
                    input_g_2 = Input(shape=(self.nn_arch['d_gene_exp'],), dtype='float32', name='input_g_2')
                    input_c_2 = Input(shape=(self.nn_arch['d_cell_type'],), dtype='float32', name='input_c_2')
                    input_A_2 = Input(shape=(self.nn_arch['n_node'], self.nn_arch['n_node'])
                                      , dtype='float32', name='input_A_2')

                    # Feature extractor.
                    outputs = CMAPFeatureExtractor(self.conf, A, name='extractor')((input_t_1, input_g_1, input_c_1, input_A_1
                                                                                       , input_t_2, input_g_2, input_c_2, input_A_2))

                    opt = optimizers.Adam(lr=self.hps['lr']
                                          , beta_1=self.hps['beta_1']
                                          , beta_2=self.hps['beta_2']
                                          , decay=self.hps['decay'])

                    self.model = Model(inputs=[input_t_1, input_g_1, input_c_1, input_A_1
                                        , input_t_2, input_g_2, input_c_2, input_A_2]
                                       , outputs=outputs)
                    self.model.compile(optimizer=opt
                                  , loss=MoA_loss
                                  , loss_weights=self.hps['loss_weights']
                                  , run_eagerly=True)
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
                                                , custom_objects={'MoA_loss': MoA_loss
                                                , 'MoA_metric': MoA_metric
                                                , 'CMAPFeatureGAE': CMAPFeatureGAE
                                                , 'CMAPFeatureExtractor': CMAPFeatureExtractor}
                                            , compile=False))
                        self.k_fold_models[i].compile(optimizer=opt
                                           , loss=MoA_loss
                                           , loss_weights=self.hps['loss_weights']
                                           , run_eagerly=True)
                else:
                    # Create models for K-fold.
                    opt = optimizers.Adam(lr=self.hps['lr']
                                          , beta_1=self.hps['beta_1']
                                          , beta_2=self.hps['beta_2']
                                          , decay=self.hps['decay'])

                    for i in range(self.nn_arch['k_fold']):
                        # Design the MoA prediction model.
                        input_t_1 = Input(shape=(self.nn_arch['d_treatment_type'],), dtype='float32', name='input_t_1')
                        input_g_1 = Input(shape=(self.nn_arch['d_gene_exp'],), dtype='float32', name='input_g_1')
                        input_c_1 = Input(shape=(self.nn_arch['d_cell_type'],), dtype='float32', name='input_c_1')
                        input_A_1 = Input(shape=(self.nn_arch['n_node'], self.nn_arch['n_node'])
                                          , dtype='float32', name='input_A_1')

                        input_t_2 = Input(shape=(self.nn_arch['d_treatment_type'],), dtype='float32', name='input_t_2')
                        input_g_2 = Input(shape=(self.nn_arch['d_gene_exp'],), dtype='float32', name='input_g_2')
                        input_c_2 = Input(shape=(self.nn_arch['d_cell_type'],), dtype='float32', name='input_c_2')
                        input_A_2 = Input(shape=(self.nn_arch['n_node'], self.nn_arch['n_node'])
                                          , dtype='float32', name='input_A_2')

                        # Feature extractor.
                        outputs = CMAPFeatureExtractor(self.conf, A, name='extractor')([input_t_1, input_g_1, input_c_1, input_A_1
                                        , input_t_2, input_g_2, input_c_2, input_A_2])

                        model = Model(inputs=[input_t_1, input_g_1, input_c_1, input_A_1
                                        , input_t_2, input_g_2, input_c_2, input_A_2]
                                           , outputs=outputs)
                        model.compile(optimizer=opt
                                           , loss=MoA_loss
                                           , loss_weights=self.hps['loss_weights']
                                           , run_eagerly=True)
                        model.summary()

                        self.k_fold_models.append(model)
            else:
                raise ValueError('cv_type is not valid.')
        elif self.conf['model_type'] == MODEL_TYPE_GAE:
            if self.model_loading:
                self.model = load_model(self.MODEL_GAE_PATH + '.h5'
                                            , custom_objects={'MoA_loss': MoA_loss
                                            , 'MoA_metric': MoA_metric
                                            , 'CMAPFeatureGAE': CMAPFeatureGAE
                                            , 'CMAPFeatureExtractor': CMAPFeatureExtractor}
                                        , compile=False)
                opt = optimizers.Adam(lr=self.hps['lr']
                                      , beta_1=self.hps['beta_1']
                                      , beta_2=self.hps['beta_2']
                                      , decay=self.hps['decay'])
                self.model.compile(optimizer=opt
                              , loss=MoA_loss
                              , loss_weights=self.hps['loss_weights']
                              , run_eagerly=True)
            else:
                # Design the MoA prediction model.
                # Input.
                input_t_1 = Input(shape=(self.nn_arch['d_treatment_type'],), dtype='float32', name='input_t_1')
                input_g_1 = Input(shape=(self.nn_arch['d_gene_exp'],), dtype='float32', name='input_g_1')
                input_c_1 = Input(shape=(self.nn_arch['d_cell_type'],), dtype='float32', name='input_c_1')

                input_A_1 = Input(shape=(self.nn_arch['n_node'], self.nn_arch['n_node'])
                                  , dtype='float32', name='input_A_1')


                # Feature extractor.
                gae = CMAPFeatureGAE(self.conf, name='gae')
                X, X_h = gae([input_t_1, input_g_1, input_c_1, input_A_1])

                # Metric loss.
                X_loss = tf.sqrt(tf.square(X_h - X))
                #A_loss = tf.sqrt(tf.square(A_h - tf.cast(A, dtype=tf.float32)))

                outputs = [X_loss] #, A_loss]

                opt = optimizers.Adam(lr=self.hps['lr']
                                      , beta_1=self.hps['beta_1']
                                      , beta_2=self.hps['beta_2']
                                      , decay=self.hps['decay'])

                self.model = Model(inputs=[input_t_1, input_g_1, input_c_1, input_A_1]
                                   , outputs=outputs)
                self.model.compile(optimizer=opt
                              , loss=MoA_loss
                              , loss_weights=self.hps['loss_weights']
                              , run_eagerly=True)
                self.model.summary()
        else:
            raise ValueError('model_type is not valid.')

        # Create dataset.
        if self.conf['model_type'] == MODEL_TYPE_GAE:
            self._create_graph_autoencoder_dataset()
        elif self.conf['model_type'] == MODEL_TYPE_EXTRACTOR:
            self._create_extractor_dataset()
        else:
            raise ValueError('model_type is not valid.')

    def _create_graph_autoencoder_dataset(self):
        input_df = pd.read_csv(os.path.join(self.raw_data_path, 'train_features.csv')).iloc[:1024]
        input_df.cp_type = input_df.cp_type.astype('category')
        input_df.cp_type = input_df.cp_type.cat.rename_categories(range(len(input_df.cp_type.cat.categories)))
        input_df.cp_time = input_df.cp_time.astype('category')
        input_df.cp_time = input_df.cp_time.cat.rename_categories(range(len(input_df.cp_time.cat.categories)))
        input_df.cp_dose = input_df.cp_dose.astype('category')
        input_df.cp_dose = input_df.cp_dose.cat.rename_categories(range(len(input_df.cp_dose.cat.categories)))

        # Remove samples of ctl_vehicle.
        valid_indexes = input_df.cp_type == 1
        input_df = input_df[valid_indexes]

        target_scored_df = pd.read_csv(os.path.join(self.raw_data_path, 'train_targets_scored.csv')).iloc[:1024]
        target_scored_df = target_scored_df[valid_indexes]
        del target_scored_df['sig_id']
        target_scored_df.columns = range(len(target_scored_df.columns))

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

            # Adjacency matrix.
            A_ = tf.cast(A, dtype='float32')

            return (tf.expand_dims(treatment_type, axis=-1), gene_exps, cell_vs, A_)

        train_val_index = np.arange(len(input_df))
        np.random.shuffle(train_val_index)
        num_val = int(self.conf['val_ratio'] * len(input_df))
        num_tr = len(input_df) - num_val
        train_index = train_val_index[:num_tr]
        val_index = train_val_index[num_tr:]
        self.train_index = train_index
        self.val_index = val_index

        # Training dataset.
        input_dataset = tf.data.Dataset.from_tensor_slices(input_df.iloc[train_index].to_dict('list'))
        input_dataset = input_dataset.map(make_input_features)

        dummy_target_dataset_1 = tf.data.Dataset.range(num_tr)
        dummy_target_dataset_2 = tf.data.Dataset.range(num_tr)

        f_target_dataset = tf.data.Dataset.zip((dummy_target_dataset_1, dummy_target_dataset_2))

        # Inputs and targets.
        tr_dataset = tf.data.Dataset.zip((input_dataset, f_target_dataset))
        tr_dataset = tr_dataset.shuffle(buffer_size=self.hps['batch_size'] * 5
                                        , reshuffle_each_iteration=True).repeat().batch(self.hps['batch_size'])
        self.step = len(input_df.iloc[train_index]) // self.hps['batch_size']

        # Validation dataset.
        input_dataset = tf.data.Dataset.from_tensor_slices(input_df.iloc[val_index].to_dict('list'))
        input_dataset = input_dataset.map(make_input_features)

        dummy_target_dataset_1 = tf.data.Dataset.range(num_val)
        dummy_target_dataset_2 = tf.data.Dataset.range(num_val)

        f_target_dataset = tf.data.Dataset.zip((dummy_target_dataset_1, dummy_target_dataset_2))

        # Inputs and targets.
        val_dataset = tf.data.Dataset.zip((input_dataset, f_target_dataset)).batch(self.hps['batch_size'])

        self.trval_dataset_autoencoder = (tr_dataset, val_dataset)

    def _create_extractor_dataset(self):
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

        target_scored_df = pd.read_csv(os.path.join(self.raw_data_path, 'train_targets_scored.csv')) #.iloc[:1024]
        target_scored_df = target_scored_df[valid_indexes]

        target_nonscored_df = pd.read_csv(os.path.join(self.raw_data_path, 'train_targets_nonscored.csv')) #.iloc[:1024]
        target_nonscored_df = target_nonscored_df[valid_indexes]

        target_df = pd.concat([target_scored_df, target_nonscored_df.iloc[:, 1:]], axis=1)
        del target_df['sig_id']
        target_df.columns = range(len(target_df.columns))

        # Get same MoA pairs.
        s_pairs = []
        target_df_sum = target_df.sum()
        n_ref = int(target_df_sum.mean())
        n_samples = comb(n_ref, 2, exact=True)

        for c in tqdm(target_df.columns):
            series = target_df[c]
            series = series[series == 1]
            idxes = list(series.index)
            if len(idxes) == 0:
                continue
            s_pair = np.random.choice(idxes, size=(n_samples, 2), replace=True)
            s_pairs.append(s_pair)

        s_pairs = np.concatenate(s_pairs, axis=0)

        def _make_input_features(pair):
            pair = pair.numpy()
            idx_1 = pair[0]
            idx_2 = pair[1]
            input_1 = input_df.loc[idx_1]
            input_2 = input_df.loc[idx_2]

            # Treatment.
            cp_time_1 = input_1['cp_time']
            cp_dose_1 = input_1['cp_dose']

            treatment_type_1 = cp_time_1 * 2 + cp_dose_1

            # Gene expression.
            gene_exps_1 = input_1.iloc[4:(4 + self.nn_arch['d_gene_exp'])].values

            # Cell viability.
            cell_vs_1 = input_1.iloc[(4 + self.nn_arch['d_gene_exp']):].values

            # Adjacency matrix.
            A_1 = A

            # Treatment.
            cp_time_2 = input_2['cp_time']
            cp_dose_2 = input_2['cp_dose']

            treatment_type_2 = cp_time_2 * 2 + cp_dose_2

            # Gene expression.
            gene_exps_2 = input_2.iloc[4:(4 + self.nn_arch['d_gene_exp'])].values

            # Cell viability.
            cell_vs_2 = input_2.iloc[(4 + self.nn_arch['d_gene_exp']):].values

            # Adjacency matrix.
            A_2 = A

            return np.expand_dims(treatment_type_1, axis=-1) \
                , gene_exps_1 \
                , cell_vs_1 \
                , A_1 \
                , np.expand_dims(treatment_type_2, axis=-1) \
                , gene_exps_2 \
                , cell_vs_2 \
                , A_2

        def make_input_features(inputs):
            t_1, g_1, c_1, A_1, t_2, g_2, c_2, A_2 = tf.py_function(_make_input_features, [inputs],
                                                                    Tout=[tf.float32] * 8)
            return t_1, g_1, c_1, A_1, t_2, g_2, c_2, A_2

        if self.conf['cv_type'] == CV_TYPE_TRAIN_VAL_SPLIT:
            train_val_index = np.arange(len(s_pairs))
            np.random.shuffle(train_val_index)
            num_val = int(self.conf['val_ratio'] * len(s_pairs))
            num_tr = len(s_pairs) - num_val
            train_index = train_val_index[:num_tr]
            val_index = train_val_index[num_tr:]
            self.train_index = train_index
            self.val_index = val_index

            # Training dataset.
            input_dataset = tf.data.Dataset.from_tensor_slices(s_pairs[self.train_index])
            input_dataset = input_dataset.map(make_input_features)

            dummy_target_dataset_1 = tf.data.Dataset.range(num_tr)
            dummy_target_dataset_2 = tf.data.Dataset.range(num_tr)

            f_target_dataset = tf.data.Dataset.zip((dummy_target_dataset_1, dummy_target_dataset_2))

            # Inputs and targets.
            tr_dataset = tf.data.Dataset.zip((input_dataset, f_target_dataset))
            tr_dataset = tr_dataset.shuffle(buffer_size=self.hps['batch_size'] * 5
                                            , reshuffle_each_iteration=True).repeat().batch(self.hps['batch_size'])
            self.step = len(s_pairs[self.train_index]) // self.hps['batch_size']

            # Validation dataset.
            input_dataset = tf.data.Dataset.from_tensor_slices(s_pairs[self.val_index])
            input_dataset = input_dataset.map(make_input_features)

            dummy_target_dataset_1 = tf.data.Dataset.range(num_val)
            dummy_target_dataset_2 = tf.data.Dataset.range(num_val)

            f_target_dataset = tf.data.Dataset.zip((dummy_target_dataset_1, dummy_target_dataset_2))

            # Inputs and targets.
            val_dataset = tf.data.Dataset.zip((input_dataset, f_target_dataset)).batch(self.hps['batch_size'])

            self.trval_extractor_dataset = (tr_dataset, val_dataset)
        elif self.conf['cv_type'] == CV_TYPE_K_FOLD: #?
            kfold = KFold(n_splits=self.nn_arch['k_fold'])
            stratified_kfold = StratifiedKFold(n_splits=self.nn_arch['k_fold'])
            # group_kfold = GroupKFold(n_splits=self.nn_arch['k_fold'])
            self.k_fold_trval_extractor_datasets = []

            for train_pairs, val_pairs in kfold.split(s_pairs):
                # Training dataset.
                input_dataset = tf.data.Dataset.from_tensor_slices(train_pairs)
                input_dataset = input_dataset.map(make_input_features)

                dummy_target_dataset_1 = tf.data.Dataset.range(len(train_pairs))
                dummy_target_dataset_2 = tf.data.Dataset.range(len(train_pairs))

                f_target_dataset = tf.data.Dataset.zip((dummy_target_dataset_1, dummy_target_dataset_2))

                # Inputs and targets.
                tr_dataset = tf.data.Dataset.zip((input_dataset, f_target_dataset))
                tr_dataset = tr_dataset.shuffle(buffer_size=self.hps['batch_size'] * 5
                                                , reshuffle_each_iteration=True).repeat().batch(self.hps['batch_size'])
                self.step = len(train_pairs) // self.hps['batch_size']

                # Validation dataset.
                input_dataset = tf.data.Dataset.from_tensor_slices(val_pairs)
                input_dataset = input_dataset.map(make_input_features)

                dummy_target_dataset_1 = tf.data.Dataset.range(len(val_pairs))
                dummy_target_dataset_2 = tf.data.Dataset.range(len(val_pairs))

                f_target_dataset = tf.data.Dataset.zip((dummy_target_dataset_1, dummy_target_dataset_2))

                # Inputs and targets.
                val_dataset = tf.data.Dataset.zip((input_dataset, f_target_dataset)).batch(self.hps['batch_size'])

                trval_extractor_dataset = (tr_dataset, val_dataset)
                self.k_fold_trval_extractor_datasets.append(trval_extractor_dataset)
        else:
            raise ValueError('cv_type is not valid.')

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

        '''
        def schedule_lr(e_i):
            self.hps['lr'] = self.hps['reduce_lr_factor'] * self.hps['lr']
            return self.hps['lr']

        lr_scheduler = LearningRateScheduler(schedule_lr, verbose=1)
        '''

        if self.conf['model_type'] == MODEL_TYPE_EXTRACTOR:
            if self.conf['cv_type'] == CV_TYPE_TRAIN_VAL_SPLIT:
                model_check_point = ModelCheckpoint(self.MODEL_PATH + '.h5'
                                                    , monitor='val_loss'
                                                    , verbose=1
                                                    , save_best_only=True)

                self.model.fit_generator(self.trval_extractor_dataset[0]
                                                    , steps_per_epoch=self.step
                                                    , epochs=self.hps['epochs']
                                                    , verbose=1
                                                    , max_queue_size=80
                                                    , workers=4
                                                    , use_multiprocessing=False
                                                    , callbacks=[model_check_point, reduce_lr] #, tensorboard]
                                                    , validation_data=self.trval_extractor_dataset[1]
                                                    , validation_freq=1
                                                    , shuffle=True)
            elif self.conf['cv_type'] == CV_TYPE_K_FOLD:
                for i in range(self.nn_arch['k_fold']):
                    model_check_point = ModelCheckpoint(self.MODEL_PATH + '_' + str(i) + '.h5'
                                                        , monitor='val_loss'
                                                        , verbose=1
                                                        , save_best_only=True)

                    self.k_fold_models[i].fit_generator(self.k_fold_trval_extractor_datasets[i][0]
                                                        , steps_per_epoch=self.step
                                                        , epochs=self.hps['epochs']
                                                        , verbose=1
                                                        , max_queue_size=80
                                                        , workers=4
                                                        , use_multiprocessing=False
                                                        , callbacks=[model_check_point, reduce_lr] #, tensorboard]
                                                        , validation_data=self.k_fold_trval_extractor_datasets[i][1]
                                                        , validation_freq=1
                                                        , shuffle=True)
            else:
                raise ValueError('cv_type is not valid.')
        elif self.conf['model_type'] == MODEL_TYPE_GAE:
            model_check_point = ModelCheckpoint(self.MODEL_GAE_PATH + '.h5'
                                                , monitor='val_loss'
                                                , verbose=1
                                                , save_best_only=True)

            self.model.fit_generator(self.trval_dataset_autoencoder[0]
                                                , steps_per_epoch=self.step
                                                , epochs=self.hps['epochs']
                                                , verbose=1
                                                , max_queue_size=80
                                                , workers=4
                                                , use_multiprocessing=False
                                                , callbacks=[model_check_point, reduce_lr] #, tensorboard]
                                                , validation_data=self.trval_dataset_autoencoder[1]
                                                , validation_freq=1
                                                , shuffle=True)
        else:
            raise ValueError('model_type is not valid.')

        #print('Save the model.')
        #self.model.save(self.MODEL_PATH, save_format='h5')
        # self.model.save(self.MODEL_PATH, save_format='tf')

    def evaluate(self):
        """Evaluate."""
        assert self.conf['cv_type'] == CV_TYPE_TRAIN_VAL_SPLIT

        input_df = pd.read_csv(os.path.join(self.raw_data_path, 'train_features.csv')).iloc[:1024]
        input_df.cp_type = input_df.cp_type.astype('category')
        input_df.cp_type = input_df.cp_type.cat.rename_categories(range(len(input_df.cp_type.cat.categories)))
        input_df.cp_time = input_df.cp_time.astype('category')
        input_df.cp_time = input_df.cp_time.cat.rename_categories(range(len(input_df.cp_time.cat.categories)))
        input_df.cp_dose = input_df.cp_dose.astype('category')
        input_df.cp_dose = input_df.cp_dose.cat.rename_categories(range(len(input_df.cp_dose.cat.categories)))

        # Remove samples of ctl_vehicle.
        input_df = input_df[input_df.cp_type == 1]

        target_scored_df = pd.read_csv(os.path.join(self.raw_data_path, 'train_targets_scored.csv')).iloc[:1024]
        target_scored_df = target_scored_df.iloc[self.val_index]
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

            # Adjacency matrix.
            A_ = tf.cast(A, dtype='float32')

            return (tf.expand_dims(treatment_type, axis=-1), gene_exps, cell_vs, A_)

        # Validation dataset.
        val_dataset = tf.data.Dataset.from_tensor_slices(input_df.iloc[self.val_index].to_dict('list'))
        val_dataset = val_dataset.map(make_input_features)

        val_iter = val_dataset.as_numpy_iterator()

        # Predict MoAs.
        gae = self.model.get_layer('gae')
        sig_id_list = []
        MoAs = [[] for _ in range(len(MoA_annots))]

        for i, d in tqdm(enumerate(val_iter)):
            t, g, c = d
            id = target_scored_df['sig_id'].iloc[i]
            t = np.expand_dims(t, axis=0)
            g = np.expand_dims(g, axis=0)
            c = np.expand_dims(c, axis=0)
            A_ = A

            if self.conf['cv_type'] == CV_TYPE_TRAIN_VAL_SPLIT:
                # First layer.
                # X.
                t = gae.embed_treatment_type_0(t)
                t = tf.reshape(t, (-1, self.nn_arch['d_input_feature']))
                t = gae.dense_treatment_type_0(t)
                # t = self.batch_normalization_0_1(t)

                g = gae.dense_gene_exp_0(g)
                # g = self.batch_normalization_0_2(g)

                c = gae.dense_cell_type_0(c)
                # c = self.batch_normalization_0_3(c)

                t = gae.layer_normalization_0_1(t)
                g = gae.layer_normalization_0_2(g)
                c = gae.layer_normalization_0_3(c)

                t = tf.expand_dims(t, axis=1)
                g = tf.expand_dims(g, axis=1)
                c = tf.expand_dims(c, axis=1)

                X = gae.concat_0([t, g, c])

                # GCN encoding layer.
                Z, _ = gae.gcn_encoder_1([X, A_])

                for i, MoA in enumerate(result):
                    MoAs[i].append(MoA)
            elif self.conf['cv_type'] == CV_TYPE_K_FOLD:
                # Conduct ensemble prediction.
                result_list = []

                for i in range(self.nn_arch['k_fold']):
                    _, _, result = self.k_fold_models[i].predict([t, g, c])
                    result = np.squeeze(result, axis=0)
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

        target_scored_df = pd.read_csv(os.path.join(self.raw_data_path, 'train_targets_scored.csv'))
        MoA_annots = target_scored_df.columns[1:]

        def make_input_features(inputs):
            # Treatment.
            id = inputs['sig_id']
            cp_type = inputs['cp_type']
            cp_time = inputs['cp_time']
            cp_dose = inputs['cp_dose']

            cp_type_onehot = tf.one_hot(cp_type, 2)
            cp_time_onehot = tf.one_hot(cp_time, 3)
            cp_dose_onehot = tf.one_hot(cp_dose, 2)

            treatment_onehot = tf.concat([cp_type_onehot, cp_time_onehot, cp_dose_onehot], axis=-1)

            # Gene expression.
            gene_exps = [inputs['g-' + str(v)] for v in range(self.nn_arch['num_gene_exp'])]
            gene_exps = tf.stack(gene_exps, axis=0)

            # Cell viability.
            cell_vs = [inputs['c-' + str(v)] for v in range(self.nn_arch['num_cell_type'])]
            cell_vs = tf.stack(cell_vs, axis=0)

            return (id, treatment_onehot, gene_exps, cell_vs)

        test_dataset = tf.data.Dataset.from_tensor_slices(input_df.to_dict('list'))
        test_dataset = test_dataset.map(make_input_features)
        test_iter = test_dataset.as_numpy_iterator()

        # Predict MoAs.
        sig_id_list = []
        MoAs = [[] for _ in range(len(MoA_annots))]

        for id, t, g, c in tqdm(test_iter):
            id = id.decode('utf8') #?
            t = np.expand_dims(t, axis=0)
            g = np.expand_dims(g, axis=0)
            c = np.expand_dims(c, axis=0)

            if self.conf['cv_type'] == CV_TYPE_TRAIN_VAL_SPLIT:
                _, _, result = self.model.layers[-1]([t, g, c]) #self.model.predict([t, g, c])
                result = np.squeeze(result, axis=0)

                for i, MoA in enumerate(result):
                    MoAs[i].append(MoA)
            elif self.conf['cv_type'] == CV_TYPE_K_FOLD:
                # Conduct ensemble prediction.
                result_list = []

                for i in range(self.nn_arch['k_fold']):
                    _, _, result = self.k_fold_models[i].predict([t, g, c])
                    result = np.squeeze(result, axis=0)
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


def main():
    """Main."""
    np.random.seed(1024)
    tf.random.set_seed(1024)

    # Load configuration.
    with open("MoA_pred_gnn_conf.json", 'r') as f:
        conf = json.load(f)

    if conf['mode'] == 'train':
        # Train.
        model = MoAPredictorGNN(conf)

        ts = time.time()
        model.train()
        te = time.time()

        print('Elasped time: {0:f}s'.format(te - ts))
    elif conf['mode'] == 'evaluate':
        # Train.
        model = MoAPredictorGNN(conf)

        ts = time.time()
        model.evaluate()
        te = time.time()

        print('Elasped time: {0:f}s'.format(te - ts))
    elif conf['mode'] == 'test':
        model = MoAPredictorGNN(conf)

        ts = time.time()
        model.test()
        te = time.time()

        print('Elasped time: {0:f}s'.format(te - ts))
    else:
        raise ValueError('Mode is not valid.')


if __name__ == '__main__':
    main()