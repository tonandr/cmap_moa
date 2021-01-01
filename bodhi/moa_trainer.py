'''
Created on 2020. 12. 6.

@author: Inwoo Chung (gutomitai@gmail.com)
'''

import numpy as np
import pandas as pd
import os
from abc import ABC, abstractmethod
import time
import json
import platform
import warnings

from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.utils import CustomObjectScope

from bodhi import MoAPredictor

# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

warnings.filterwarnings('ignore')

# Constants.
DEBUG = True

TRAINING_STRATEGY_MAX = 'max'
TRAINING_STRATEGY_CONTINUE = 'continue'

epsilon = 1e-7


def create_scaling_func(a, b):
    return lambda x: (b - a) * x + a


def create_log_scaling_func(a, b):
    return lambda x: 10 ** ((np.log10(b + epsilon) - np.log10(a + epsilon)) * x + np.log10(a + epsilon))


def policy_loss(y_true, y_pred):
    return K.mean(y_pred)


class Critic(ABC):
    """Abstract critic class."""

    @abstractmethod
    def __init__(self, hps, resource_path, model_loading, *args, **keywords):
        pass

    @abstractmethod
    def train(self, state, action, td_target):
        pass

    @abstractmethod
    def predict_action_value(self, state, action):
        pass


class Actor(ABC):
    """Abstract actor class."""

    @abstractmethod
    def __init__(self, hps, resource_path, model_loading, *args, **keywords):
        pass

    @abstractmethod
    def train(self, state, action, td_error):
        pass

    @abstractmethod
    def act(self, state):
        pass


class RLModel(ABC):
    """Abstract reinforcement learning model."""

    @abstractmethod
    def __init__(self, config_path, resource_path, model_loading, *args, **keywords):
        pass

    @abstractmethod
    def learn(self, *args, **keywords):
        pass

    @abstractmethod
    def act(self, *args, **keywords):
        pass


class Learner():
    """Abstract learner."""
    pass


class Trainer():
    """Abstract trainer."""
    pass


class ModelTrainer(Trainer):
    """Model optimization via RL."""

    class OptCritic(Critic):
        """Critic."""

        # Constants.
        MODEL_PATH = 'opt_critic'

        def __init__(self, resource_path, conf):
            """
            Parameters
            ----------
            resource_path: String.
                Resource path.
            conf: Dictionary.
                Configuration.
            """

            # Initialize.
            self.resource_path = resource_path
            self.conf = conf
            self.hps = conf['hps']
            self.nn_arch = conf['nn_arch']
            self.model_loading = conf['model_loading']

            if self.model_loading:
                self.model = load_model(os.path.join(self.MODEL_PATH))  # Check exception.
            else:
                # Design action value function.
                # Input.
                input_a = Input(shape=(self.nn_arch['action_dim'],), name='input_a')

                # Get action value.
                x = input_a
                for i in range(self.nn_arch['num_layers']):
                    x = Dense(self.nn_arch['dense_layer_dim'], activation='relu', name='dense' + str(i + 1))(x)

                action_value = Dense(1, activation='linear', name='action_value_dense')(x)

                self.model = Model(inputs=[input_a], outputs=[action_value], name='opt_critic')

                opt = optimizers.Adam(lr=self.hps['lr']
                                      , beta_1=self.hps['beta_1']
                                      , beta_2=self.hps['beta_2']
                                      , decay=self.hps['decay'])

                self.model.compile(optimizer=opt, loss='mse', loss_weights=[1])
                # self.model.summary()

        def train(self, a, td_target):  # learning rate?
            """Train critic.

            Parameters
            ----------
            a: 2D numpy array.
                Action, a.
            td_target : 2D numpy array.
                TD target array, batch size (value)?
            """

            # Train model online.
            self.model.train_on_batch([a], [td_target])

            # Save the model.
            #self.model.save(os.path.join(self.MODEL_PATH))

        def predict_action_value(self, a):
            """Predict action value.

            Parameters
            ----------
            a: 2D numpy array.
                Action, a.

            Returns
            -------
            Action value.
                2D numpy array.
            """

            # Predict action value.
            action_value = self.model.predict([a])

            return action_value

    class OptActor(Actor):
        """Actor."""

        # Constants.
        MODEL_PATH = 'opt_actor'

        def __init__(self, resource_path, conf):
            """
            Parameters
            ----------
            resource_path: String.
                Raw data path.
            conf: Dictionary.
                Configuration.
            """

            # Initialize.
            self.resource_path = resource_path
            self.conf = conf
            self.hps = conf['hps']
            self.nn_arch = conf['nn_arch']
            self.model_loading = conf['model_loading']

            opt = optimizers.Adam(lr=self.hps['lr']
                                  , beta_1=self.hps['beta_1']
                                  , beta_2=self.hps['beta_2']
                                  , decay=self.hps['decay'])

            if self.model_loading:
                with CustomObjectScope({'policy_loss': policy_loss}):
                    self.model = load_model(os.path.join(self.MODEL_PATH))  # Check exception.
            else:
                # Design actor.
                # Input.
                input_s = Input(shape=(self.nn_arch['state_dim'],), name='input_s')
                input_a = Input(shape=(self.nn_arch['action_dim'],), name='input_a')
                input_old_a = Input(shape=(1,), name='input_old_a')

                # Get action.
                x = input_s
                for i in range(self.nn_arch['num_layers']):
                    x = Dense(self.nn_arch['dense_layer_dim'], activation='relu', name='dense' + str(i + 1))(x)

                x = Dense(self.nn_arch['dense1_dim']
                                       , activation='relu'
                                       , name='dense_p')(x)
                x_mu = Dense(self.nn_arch['dense1_dim']
                          , activation='relu'
                          , name='dense_mu1')(x)
                x_sigma = Dense(self.nn_arch['dense1_dim']
                             , activation='relu'
                             , name='dense_sigma1')(x)
                mu = Dense(self.nn_arch['action_dim'], name='dense_mu2')(x_mu)
                sigma = Dense(self.nn_arch['action_dim'], activation='softplus', name='dense_sigma2')(x_sigma)
                action = Lambda(lambda x: K.exp(-1.0 * K.square(x[0] - x[1]) / (2.0 * K.square(x[2]) + epsilon)) \
                                          / (x[2] * np.sqrt(2.0 * np.pi) + epsilon))([input_a, mu, sigma])

                input_td_error = Input(shape=(1,))
                r = Lambda(lambda x: x[0] / (x[1] + epsilon))([action, input_old_a])
                l1 = Lambda(lambda x: x[0] * x[1])([input_td_error, r])
                epsilon_p = self.nn_arch['epsilon_p']
                l2 = Lambda(lambda x: K.clip(x
                                            , 1.0 - epsilon_p
                                            , 1.0 + epsilon_p))(r)
                l2 = Lambda(lambda x: x[0] * x[1])([input_td_error, l2])
                l2 = Lambda(lambda x: x[0] * x[1])([input_td_error, l2])
                output = Lambda(lambda x: K.minimum(x[0], x[1]))([l1, l2])

                self.model = Model(inputs=[input_s, input_a, input_old_a, input_td_error], outputs=[output])
                self.model.compile(optimizer=opt, loss=policy_loss, loss_weights=[-1])
                # self.model.summary()

            self._make_action_model()

        def _make_action_model(self):
            """Make action model."""
            input_s = Input(shape=(self.nn_arch['state_dim'],), name='input_s')

            # Get action.
            x = input_s
            for i in range(self.nn_arch['num_layers']):
                x = self.model.get_layer('dense' + str(i + 1))(x)

            x = self.model.get_layer('dense_p')(x)

            mu = self.model.get_layer('dense_mu1')(x)
            mu = self.model.get_layer('dense_mu2')(mu)

            sigma = self.model.get_layer('dense_sigma1')(x)
            sigma = self.model.get_layer('dense_sigma2')(sigma)

            self.action_model = Model(inputs=[input_s], outputs=[mu, sigma])

        def train(self, s, a, old_a, td_error):
            """Train actor.

            Parameters
            ----------
            s: 2D numpy array.
                State, s.
            a: 2D numpy array.
                Action, a.
            old_a: 2D numpy array.
                Old action, old_a.
            td_error: 2D numpy array.
                TD error values.
            """

            # Train.
            self.model.train_on_batch([s, a, old_a, td_error]  # td_error dimension?
                                          , [a])

            # Save the model.
            #self.model.save(os.path.join(self.MODEL_PATH))

        def act(self, s):
            """Get hyper-parameters and neural network architecture information.

            Parameters
            ----------
            s: 2D numpy array.
                State, s.

            Returns
            ------
            2D numpy array.
                Bound model configuration.
            """

            # Sample actions using the normal random sampling action function.
            mu, sigma = self.action_model.predict(s)

            return np.random.normal(loc=mu, scale=sigma / 5.0)

    def __init__(self, config_path):
        """
        Parameters
        ----------
        config_path: String.
            Configuration file path.
        """

        # Load configuration.
        with open(os.path.join(config_path), 'r') as f:
            self.conf = json.load(f)

        self.resource_path = self.conf['resource_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']
        self.critic_conf = self.conf['critic_conf']
        self.actor_conf = self.conf['actor_conf']

        # Instantiate critic, actor.
        self.critic = self.OptCritic(self.resource_path, self.critic_conf)
        self.actor = self.OptActor(self.resource_path, self.actor_conf)
        self.old_actor = self.OptActor(self.resource_path, self.actor_conf)
        self.old_actor.action_model.set_weights(self.actor.action_model.get_weights())

        # Initial state and action.
        self.state = np.random.normal(
            size=(self.hps['batch_size'], self.nn_arch['state_dim']))  # Optimal initializer. ?
        self.action = self.act(self.state)
        self.old_action = self.old_act(self.state)
        self.avg_reward = np.zeros((self.hps['batch_size'], 1), dtype='float32')

    def learn(self, feedback):
        """Learn."""

        # Train critic and actor for reward and state.
        # Get rewards, states.
        state_p = feedback['state']
        reward = feedback['reward']

        # Sample next action.
        action_p = self.actor.act(state_p)

        # Train. Dimension?
        td_target = reward - self.avg_reward + self.critic.predict_action_value(action_p)
        td_error = td_target - self.critic.predict_action_value(self.action)

        self.critic.train(self.action, td_target)
        self.actor.train(self.state, self.action, self.old_action, td_error)

        self.state = state_p
        self.old_action = self.old_act(self.state)
        self.action = action_p
        self.avg_reward = self.avg_reward + self.hps['eta'] * td_error

        self.old_actor.action_model.set_weights(self.actor.action_model.get_weights())

    def act(self, s):  # Both same function?
        """Get a weight value.

        Parameters
        ----------
        s: 2D numpy array.
            State, s.

        Returns
        ------
        Float32.
            A weight value.
        """
        return self.actor.act(s)

    def old_act(self, s):  # Both same function?
        """Get a weight value.

        Parameters
        ----------
        s: 2D numpy array.
            State, s.

        Returns
        ------
        Float32.
            A weight value.
        """
        return self.old_actor.act(s)

    def _tanh(self, x):
        return (np.exp(x) - np.exp(-1.0 * x)) / (np.exp(x) + np.exp(-1.0 * x) + epsilon)

    def optimize(self, f_conf):
        """Optimize the model via RL."""
        tr_res_df = pd.DataFrame(columns=['conf', 'loss'])
        rs_min = -1.0

        # Create scaling functions for each parameter.
        s_funcs = []
        #s_funcs.append(create_scaling_func(2.0, 6.0))  # batch_size.
        s_funcs.append(create_log_scaling_func(1e-3, 1e-2))  # hps: ls.
        s_funcs.append(create_log_scaling_func(1e-4, 1e-3))  # hps: weight_decay.
        s_funcs.append(create_scaling_func(0.2, 0.3))  # nn_arch: dropout_rate.
        s_funcs.append(create_scaling_func(16, 128))  # nn_arch: d_cv_init.

        for i in tqdm(range(self.hps['steps'])):
            # Convert normalized hyper-parameters and NN architecture information to original values.
            action = (self._tanh(self.action) + 1.0) * 0.5  # -1.0 ~ 1.0 -> 0.0 ~ 1.0.
            action = action.ravel()
            rs = []

            for j in range(self.hps['batch_size']): #?
                # Convert normalized hyper-parameters and NN architecture information to original values.
                hp_norm = action  # 0.0 ~ 1.0.

                # hps.
                # f_conf['hps']['batch_size'] = int(s_funcs[0](hp_norm[0]))
                f_conf['hps']['ls'] = s_funcs[0](hp_norm[0])
                f_conf['hps']['weight_decay'] = s_funcs[1](hp_norm[1])

                f_conf['nn_arch']['dropout_rate'] = s_funcs[2](hp_norm[2])
                f_conf['nn_arch']['d_cv_init'] = int(s_funcs[3](hp_norm[3]))
                f_conf['nn_arch']['d_get_init'] = int(8 * s_funcs[3](hp_norm[3]))
                f_conf['nn_arch']['d_f_init'] = int(4 * s_funcs[3](hp_norm[3]))

                # Train.
                try:
                    model = MoAPredictor(f_conf)

                    ts = time.time()
                    hist = model.train()
                    te = time.time()
                except Exception as e:
                    print(e)
                    continue

                # print('Elapsed time: {0:f}s'.format(te-ts))

                # Calculate reward.
                loss = hist.history['val_moap_2_loss'][-1]
                del model

                # Check exception.
                if np.isnan(loss):
                    continue

                rs.append(loss)

            rs = np.expand_dims(np.asarray(rs), axis=-1)

            # Check exception.
            if len(rs) == 0:
                self.state = np.random.normal(
                    size=(self.hps['batch_size'], self.nn_arch['state_dim']))  # Optimal initializer. ?
                self.action = self.act(self.state)
                continue

            if self.conf['training_strategy'] == TRAINING_STRATEGY_MAX:
                if rs.mean() > rs_min:
                    print('Save the model.')
                    self.critic.model.save(self.critic.MODEL_PATH)
                    self.actor.model.save(self.actor.MODEL_PATH)
                    rs_min = rs.mean()
            elif self.conf['training_strategy'] == TRAINING_STRATEGY_CONTINUE:
                print('Save the model.')
                self.critic.model.save(self.critic.MODEL_PATH)
                self.actor.model.save(self.actor.MODEL_PATH)
            else:
                raise ValueError('Training strategy is not valid.')

            print(f_conf, rs.mean())
            tr_res_df = tr_res_df.append(pd.DataFrame({'conf': [str(f_conf)], 'reward': [rs.mean()]}))

            self.state = np.random.normal(
                size=(self.hps['batch_size'], self.nn_arch['state_dim']))  # Optimal initializer. ?
            feedback = {'state': self.state, 'reward': rs}
            self.learn(feedback)
            self.action = self.act(self.state)

            # Save the training result.
            tr_res_df.to_csv('training_result.csv', index=False)

    def optimize_via_random_search(self, f_conf):
        """Optimize the model via Random Search."""
        tr_res_df = pd.DataFrame(columns=['conf', 'loss'])

        # Create scaling functions for each parameter.
        s_funcs = []
        #s_funcs.append(create_scaling_func(2.0, 6.0))  # batch_size.
        s_funcs.append(create_log_scaling_func(1e-3, 1e-2))  # hps: ls.
        s_funcs.append(create_log_scaling_func(1e-4, 1e-3))  # hps: weight_decay.
        s_funcs.append(create_scaling_func(0.2, 0.3))  # nn_arch: dropout_rate.
        s_funcs.append(create_scaling_func(16, 128))  # nn_arch: d_cv_init.

        for i in tqdm(range(self.hps['trials'])):
            # Convert normalized hyper-parameters and NN architecture information to original values.
            hp_norm = np.random.rand(4)  # 0.0 ~ 1.0.

            # hps.
            #f_conf['hps']['batch_size'] = int(s_funcs[0](hp_norm[0]))
            f_conf['hps']['ls'] = s_funcs[0](hp_norm[0])
            f_conf['hps']['weight_decay'] = s_funcs[1](hp_norm[1])

            f_conf['nn_arch']['dropout_rate'] = s_funcs[2](hp_norm[2])
            f_conf['nn_arch']['d_cv_init'] = int(s_funcs[3](hp_norm[3]))
            f_conf['nn_arch']['d_get_init'] = int(8 * s_funcs[3](hp_norm[3]))
            f_conf['nn_arch']['d_f_init'] = int(4 * s_funcs[3](hp_norm[3]))

            # Train.
            try:
                model = MoAPredictor(f_conf)

                ts = time.time()
                hist = model.train()
                te = time.time()
            except Exception as e:
                print(e)
                continue

            #print('Elapsed time: {0:f}s'.format(te-ts))

            # Calculate reward.
            loss = hist.history['val_moap_2_loss'][-1]
            del model

            # Check exception.
            if np.isnan(loss):
                continue

            print(f_conf, loss)
            tr_res_df = tr_res_df.append(pd.DataFrame({'conf': [str(f_conf)], 'loss': [loss]}))

            # Save the training result.
            tr_res_df.to_csv('training_result_rs.csv', index=False)

def main():
    # Optimize the model.
    seed = int(time.time())
    seed = 1024
    print(f'Seed:{seed}')
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Create the optimization RL entity.
    config_path = 'model_opt_conf.json'
    opt = ModelTrainer(config_path)

    with open("MoA_pred_conf.json", 'r') as f:
        f_conf = json.load(f)

    if opt.conf['opt_type'] == 'rl_opt':
        opt.optimize(f_conf)
    elif opt.conf['opt_type'] == 'random_search':
        opt.optimize_via_random_search(f_conf)
    else:
        raise ValueError('opt_type is not valid.')

if __name__ == '__main__':
    main()