import numpy as np
from scipy import stats
import pandas as pd
import os
import sys
from timeit import default_timer
from datetime import datetime
from time import gmtime
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from pprint import pprint

import json

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

MODELS_DIR = "models/"
LOGS_DIR = "logs/"
PLOTS_DIR = "plots/"


class Model:

    def __init__(self, simulator, init_cov=0.5):
        self.simulator = simulator
        self.x = self.simulator.sense()
        self.nr_of_states = 1

        # Build initial state
        initial_state = []

        # Add truck gps coordinates to the initial state
        for truck in self.simulator.truck_names:
            initial_state.extend(self.simulator.gps_map[truck])

        # # Add package gps coordinates to the initial state
        # for package in self.simulator.package_names:
        #     initial_state.extend(self.simulator.gps_map[package])
        #
        # initial_state.extend(self.simulator.loaded_packages)

        # Add all trucks RFID coordinates to the initial state
        all_trucks_RFID = []
        for truck in range(len(self.simulator.truck_names)):
            single_truck_RFID = []
            for package in range(len(self.simulator.package_names)):
                if self.simulator.in_vehicle(package, truck):
                    single_truck_RFID.append(1)
                else:
                    single_truck_RFID.append(0)
            all_trucks_RFID.extend(single_truck_RFID)

        # Add all locations RFID coordinates to the initial state
        all_locations_RFID = []
        for location in range(len(self.simulator.location_names)):
            single_location_RFID = []
            for package in range(len(self.simulator.package_names)):
                if self.simulator.package_at(package, location):
                    single_location_RFID.append(1)
                else:
                    single_location_RFID.append(0)
            all_locations_RFID.extend(single_location_RFID)

        initial_state.extend(all_trucks_RFID)
        initial_state.extend(all_locations_RFID)

        self.states = [initial_state]
        self.gamma = np.zeros((len(simulator.actions), 1), dtype=int) - np.ones((len(simulator.actions), 1), dtype=int)
        self.current_state = initial_state

        # self.mu = np.array([simulator.sense()])
        self.mu = np.array([initial_state])

        trucks_length = len(self.simulator.truck_names)*2

        # self.init_cov = []
        # self.init_cov = np.concatenate((init_cov[0]*np.identity(trucks_length)))
        # self.init_cov.extend(init_cov[1]*np.identity(len(initial_state)-trucks_length))

        init_cov_vector = [init_cov[0] for i in range(trucks_length)]
        init_cov_vector.extend(init_cov[1] for i in range(len(initial_state)-trucks_length))

        self.init_cov = init_cov_vector * np.identity(len(initial_state))

        self.cov = np.array([self.init_cov])
        self.perception_variables = tuple(['X' + str(i) for i in range(len(self.x))])
        self.action_labels = self.simulator.action_labels

    # Compute the likelihood of observing perception values 'x' given the state 's'
    def f(self, x):
        # all_state_likelihoods = []
        #
        # for s in range(0, len(self.states)):
        #
        #     single_state_likelihood = []
        #     for i in range(len(x)):
        #         single_state_likelihood.append(stats.multivariate_normal.pdf(x[i], self.mu[s][i], self.cov[s][i][i], allow_singular=True))
        #
        #     all_state_likelihoods.append(single_state_likelihood)
        #
        # return all_state_likelihoods

        return np.array([stats.multivariate_normal.pdf(x, self.mu[s], self.cov[s], allow_singular=True)
                         for s in range(0, len(self.states))])


    # [0.035206532676429945, 0.035206532676429945, 1.0328830949345564, 1.0328830949345564, 1.0328830949345564, 1.0328830949345564, 1.0328830949345564, 1.0328830949345564, 1.0328830949345564, 1.0328830949345564,
    #  1.0328830949345564, 1.0328830949345564, 1.0328830949345564, 1.0328830949345567, 1.0328830949345567, 1.0328830949345567, 1.0328830949345564, 1.0328830949345564, 1.0328830949345564, 1.0328830949345564,
    #  1.0328830949345564, 1.0328830949345564, 1.0328830949345564]


    def new_state(self):
        self.nr_of_states += 1

        new_state = self.x

        self.states.append(new_state)

        # self.gamma = np.concatenate((self.gamma, np.array([[len(self.states)]] * len(self.simulator.actions))), axis=1)
        self.gamma = np.concatenate((self.gamma, np.array([[-1]] * len(self.simulator.actions))), axis=1)

        self.mu = np.concatenate((self.mu, [new_state]), axis=0)
        self.cov = np.concatenate((self.cov, [self.init_cov]), axis=0)

        return self.states[-1]

    # def plot(self,show=False,save=False):
    #     if self.nr_of_states > 20 or len(self.action_labels) > 20:
    #         print("The size of the model is too big to be plotted")
    #         print("I plot only the set of states")
    #         self.plot_states(show=show,save=save)
    #         return
    #
    #     plt.figure(figsize=(1.75*(len(self.action_labels)+1),1.3*self.nr_of_states))
    #     for s in self.states:
    #         sub_s = plt.subplot(self.nr_of_states,
    #                             len(self.action_labels)+1,
    #                             s*(len(self.action_labels)+1)+1)
    #         self.simulator.plot_state(sub_s, self.mu[s].reshape(-1, 2))
    #         plt.title(str(s))
    #         effect = 1
    #         for a in range(len(self.action_labels)):
    #             if self.gamma[a,s] != s:
    #                 effect += 1
    #                 sub_gas = plt.subplot(self.nr_of_states,
    #                                 len(self.action_labels)+1,
    #                                 s*(len(self.action_labels) + 1)+effect)
    #                 self.simulator.plot_state(sub_gas,self.mu[self.gamma[a,s]].reshape(-1,2))
    #                 plt.title("{}({})={}".format(self.action_labels[a],str(s),str(self.gamma[a,s])))
    #         plt.tight_layout()
    #     if show:
    #         plt.show()
    #     if save:
    #         if not os.path.exists(PLOTS_DIR):
    #             print("creating the folder \"{}\", ".format(PLOTS_DIR), end='')
    #             os.mkdir(PLOTS_DIR)
    #         filename = PLOTS_DIR + self.simulator.name + save + ".png"
    #         print("saving the plot of the model {}".format(filename), flush=True)
    #         plt.pause(1)
    #         plt.savefig(filename)
    #
    # def plot_states(self,show=True,save=False):
    #     plt.figure(figsize=(20,0.14*self.nr_of_states))
    #     for s in self.states:
    #         sub_s = plt.subplot(self.nr_of_states//10 + 1,10,s + 1)
    #         self.simulator.plot_state(sub_s, self.mu[s].reshape(-1, 2))
    #         plt.title(str(s))
    #     plt.tight_layout()
    #     if save:
    #         if not os.path.exists(PLOTS_DIR):
    #             print("creating the folder \"{}\", ".format(PLOTS_DIR), end='')
    #             os.mkdir(PLOTS_DIR)
    #         filename = PLOTS_DIR + self.simulator.name + save + " states.png"
    #         print("saving the plot of the model {}".format(filename), flush=True)
    #         plt.savefig(filename)
    #     if show:
    #         plt.show()

    def save(self,label=""):
        if not os.path.exists(MODELS_DIR):
            print("creating the folder \"{}\", ".format(MODELS_DIR), end='')
            os.mkdir(MODELS_DIR)
        filename = MODELS_DIR+self.simulator.name+label
        print("saving model {}".format(filename), flush=True)
        f = open(filename,'wb')
        pickle.dump(self, f)
        f.close()

    def gt_state(self,s):
        mu_s = self.mu[s]
        nr_of_trucks = self.simulator.nr_of_trucks
        sample_s = mu_s.reshape(nr_of_trucks, -1)
        result = [[b] for b in range(nr_of_trucks) if sample_s[b, 1] < 0.5]
        for l in range(1, nr_of_trucks):
            for b in range(nr_of_trucks):
                if abs(sample_s[b, 1] - l) < 0.5:
                    for col in result:
                        if abs(sample_s[col[-1], 0] - sample_s[b, 0]) < .5:
                            col.append(b)
        if len(sum(result,[])) == self.simulator.nr_of_blocks:
            return {tuple(col) for col in result}
        else:
            # print("no ground truth state for",mu_s)
            return None

    def gt_apply(self,a, s):
        if s == None:
            return None
        new_s = [[b for b in c] for c in s]
        a_type = a[0].__name__
        a_args = a[1]
        if a_type == "ontable":
            for col in new_s:
                if col[-1] == a_args[0]:
                    del col[-1]
                    if len(col) == 0:
                        new_s.remove(col)
                    new_s.append([a_args[0]])
        if a_type == "puton":
            for col in new_s:
                if col[-1] == a_args[0]:
                    for col1 in new_s:
                        if col1[-1] == a_args[1]:
                            del col[-1]
                            if len(col) == 0:
                                new_s.remove(col)
                            col1.append(a_args[0])
        return {tuple(c) for c in new_s}

    def precision(self):
        good , bad = (0,0)
        states = self.states
        actions = self.simulator.actions
        for s in states:
            for a in range(len(actions)):
                gt_s = self.gt_state(s)
                gt_gamma_a_s = self.gt_state(self.gamma[a, s])
                gt_a_s = self.gt_apply(actions[a], gt_s)
                if gt_a_s is None or gt_gamma_a_s is None or gt_gamma_a_s != gt_a_s:
                    bad += 1
                else:
                    good += 1
        return good/(good+bad)

    def coverage(self):
        discovered_states = [self.gt_state(s) for s in self.states]
        discovered_states = {tuple(s) for s in discovered_states if s is not None}
        return len(discovered_states)/self.simulator.nr_of_discrete_states

    def redundancy(self):
        discovered_states = [self.gt_state(s) for s in self.states]
        discovered_states = {tuple(s) for s in discovered_states if s is not None}
        return self.nr_of_states/len(discovered_states)


def load_model(filename):
    f = open(filename, "rb")
    print("loading model {}".format(filename))
    m = pickle.load(f)
    f.close()
    return m


def my_softmax(data_vector):
    return [(np.exp(j) / np.sum(np.exp(i) for i in data_vector)) for j in data_vector]


class Learner:
    def __init__(self, model, strategy='random', epsilon=0.1, alpha=0.9, beta=0.9,
                 delta=0,eval_frequency=10,save_frequency=1000):
        self.model = model
        assert strategy == 'random' or strategy == 'complete', 'The value of strategy must be "random" or "complete"'
        self.strategy = model.simulator.complete_walk() if strategy == 'complete' else 'random'
        self.one_minus_epsilon = 1 - epsilon
        self.alpha = alpha
        self.alpha_on_one_minus_alpha = alpha/(1-alpha)
        self.beta = beta
        self.delta = delta
        self.eval_frequency = eval_frequency
        self.eval = pd.DataFrame(columns=('timestamp','iter','nr_of_states','precision','coverage','redundancy'))
        self.last_action = None
        self.last_state = None

        self.min_likelihood = (self.one_minus_epsilon)*\
                              stats.multivariate_normal.pdf(
                                  np.zeros_like(self.model.mu[0]),
                                  np.zeros_like(self.model.mu[0]),
                                  model.init_cov)

        # self.T = pd.DataFrame(columns=("source", "action", "target"))  # the history of all the transitions
        self.TRANS = defaultdict(int)
        # self.O = pd.DataFrame([np.concatenate(([model.current_state],
        #                                        model.mu[model.current_state]))],
        #                       columns=('state',) + model.perception_variables)
        self.OBS = defaultdict(list)
        self.iter = 0
        self.time_at_iter = [0.]
        self.save_frequency = save_frequency

    def select_action(self):
        """for the time being the selection of action is random. We should think and
        implement more smart action selection policies"""
        if self.strategy == "random":
            return np.random.choice(len(self.model.action_labels))
        else:
            return self.strategy[self.iter % len(self.strategy)]

# Update the transition
    def update_gamma(self):

        # If alpha == 1 ====> the agent never changes the transition gamma
        if 1 - self.alpha:
            if self.TRANS[(self.model.states.index(self.last_state),
                          self.last_action,
                          self.model.states.index(self.model.current_state))] - \
               self.TRANS[(self.model.states.index(self.last_state),
                           self.last_action,
                           self.model.gamma[self.last_action, self.model.states.index(self.last_state)])] > \
                    self.alpha_on_one_minus_alpha:
                self.model.gamma[self.last_action, self.model.states.index(self.last_state)] = \
                    self.model.states.index(self.model.current_state)


    def update_f(self):
        """
        Update the perception function of each state. The mean mu is updated, as well as the covariance cov.
        The updates are controlled by the parameter 'beta': the less is beta, the more the agent trusts in the
        last observation.
        :return:
        """
        state = self.model.current_state
        mu = self.model.mu
        cov = self.model.cov
        beta = self.beta

        state_index = self.model.states.index(state)
        obs_on_state = self.OBS[state_index]
        mu[state_index] = beta * mu[state_index] + (1 - beta) * np.mean(obs_on_state, axis=0)  # obs_on_state.mean()
        if len(obs_on_state) > 1:
            cov_obs = np.cov(obs_on_state, rowvar=False)  # obs_on_state.cov()
            cov[state_index] = beta * cov[state_index] +(1 -beta) * cov_obs

    # FARE FILE DI CONFIGURAZIONE in input, i parametri sono:
    # numero di iterazioni
    # covarianza iniziale
    # alfa: serve a controllare l'aggiornamento della funzione di transizione. Se alfa == 1 ==> non viene
    #       mai aggiornata la funzione di transizione.
    # beta: serve ad aggiornare la perception function, in particolare i parametri: media e covarianza
    # epsilon: serve a imporre la soglia sulla maximum likelihood


    def learn(self, max_iter=1000,
              eval_frequency=None,
              strategy=None,
              alpha=None,
              beta=None,
              delta=None,
              epsilon=None,
              save_frequency=None):

        m = self.model

        simulator = m.simulator
        now = default_timer()
        if self.strategy is not None:
            self.strategy = m.simulator.complete_walk() if strategy == 'complete' else 'random'
        if eval_frequency is not None:
            self.eval_frequency = eval_frequency
        if epsilon is not None:
            self.min_likelihood += self.min_likelihood * (1 - self.one_minus_epsilon - epsilon) / self.one_minus_epsilon
            self.one_minus_epsilon = 1 - epsilon
        if alpha is not None:
            self.alpha = alpha
            self.alpha_on_one_minus_alpha = alpha / (1 - alpha)
        if beta is not None:
            self.beta = beta
        if delta is not None:
            self.delta = delta
        if save_frequency is not None:
            self.save_frequency = save_frequency

        # If the min likelihood has not been already set, set the minimum likelihood to the likelihood obtained
        # after the execution of the first no-op action in the initial state.
        # no_op = 86
        #
        # # Execute the no-op
        # simulator.execute_action(no_op)
        #
        # # Perceive the noisy data
        # m.x = simulator.sense()

        # Likelihood computation
        #
        # fx = m.f(m.x)
        #
        # print("Initial no-op computed likelihood is: {}".format(fx))
        #
        # self.min_likelihood = fx[0]/2

        # ideal_x = [850, 550, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        # maximum_noise = [5, 5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, -0.2, -0.2, -0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        # # maximum_noisy_x = ideal_x + maximum_noise
        # maximum_noisy_x = [(x + y) for (x, y) in zip(ideal_x[1:], maximum_noise[1:])]

        # maximum_noisy_x = [855, 555, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        maximum_noisy_x = [855, 555, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

        fx = m.f(maximum_noisy_x)
        self.min_likelihood = fx[0]

        print("Initial single likelihood thresholds are: {}".format(self.min_likelihood))
        print("Initial global likelihood threshold is: {}".format(np.product(self.min_likelihood)))

        # Iterate for a maximum number of exploration steps
        for i in range(max_iter):

            # Evaluates online metrics and save it into a log file
            if i % self.eval_frequency == 0:
                self.eval_log()

            outcome = False

            # While the chosen action is illegal
            while not outcome:

                # Choose an action according to the strategy defined a priori.
                a = self.select_action()

                # The state goal is unreachable ??????
                if a is None:
                    print("exiting for problems")
                    self.save()
                    break

                # Execute the action through the domain simulator
                outcome = simulator.execute_action(a)

                # DEBUG
                # if outcome:
                #     print("\n\nSuccessfully executed action {}: {}".format(a, self.model.action_labels[a]))

            # Perceive
            m.x = simulator.sense()

            self.last_action = a

            # Likelihood computation
            fx = m.f(m.x)

            # DEBUG
            # print("\n\n\nComputed maximum likelihood: {}".format(fx))

            # prior = (1 + self.delta*((m.gamma[a,m.current_state] == m.states)*(m.nr_of_states-1) - \
            #         (m.gamma[a,m.current_state] != m.states)))/(1 + self.delta*(m.nr_of_states - 1))
            # prior = 1
            # best_next_state_index = np.argmax(fx*prior)




            # filtered_fx = [fx[i] for i in range(len(fx)) if fx[i] > self.min_likelihood]
            # filtered_fx = [np.product(filtered_fx[i]) for i in range(len(filtered_fx))]



            # Compute the state that maximizes the likelihood of observing x
            best_next_state_index = np.argmax(fx)
            # best_next_state_index = np.argmax(filtered_fx)

            # If the state is under the minimum likelihood threshold, then create a new state
            if fx[best_next_state_index] < self.min_likelihood:
                best_next_state = m.new_state()
                # DEBUG
                # print("New state created: state {}".format(len(self.model.states) - 1))
            else:
                best_next_state = self.model.states[best_next_state_index]
                # DEBUG
                # print("Coming back to state {}".format(best_next_state_index))

            self.last_state = m.current_state
            m.current_state = best_next_state

            # self.T.loc[len(self.T)] = [self.last_state,
            #                            self.last_action,
            #                            m.current_state]


            # self.TRANS[self.last_state, self.last_action, m.current_state] += 1
            self.TRANS[m.states.index(self.last_state), self.last_action, m.states.index(m.current_state)] += 1

            # self.O.loc[len(self.O)] = np.concatenate(([m.current_state], m.x))

            self.OBS[m.states.index(m.current_state)].append(m.x)
            self.update_gamma()
            self.update_f()
            new_now = default_timer()
            self.time_at_iter.append(new_now-now)
            now = new_now
            if i % self.save_frequency == 0 and i != 0:
                self.save()
            self.iter += 1

            # if self.iter >3:
            #     self.delete_isolated_states()

        # self.delete_isolated_states()

        self.eval_log()

        # self.save()

    def eval_log(self):
        evaluate = {'timestamp': sum(self.time_at_iter[:self.iter]),
                'iter': self.iter,
                'nr_of_states': m.nr_of_states,
                'precision': -1,
                # 'precision': m.precision(),
                'coverage': -1,
                'redundancy': -1}
        print("    ".join([str(i) for i in list(evaluate.values())]))
        self.eval = self.eval.append(evaluate, ignore_index=True)

    def save(self, filename=None):
        if not os.path.exists(LOGS_DIR):
            print("creating the folder \"{}\", ".format(LOGS_DIR), end='')
            os.mkdir(LOGS_DIR)
        now = datetime(*gmtime()[:6]).isoformat()
        model = self.model.simulator.name
        strategy = "complete" if self.strategy != "random" else "random"
        param = "iter={} alpha={} beta={} epsilon={} delta={} strat={}".format(self.iter,
                                                                               self.alpha,
                                                                               self.beta,
                                                                               1-self.one_minus_epsilon,
                                                                               self.delta,
                                                                               strategy)
        if filename is None:
            filename = LOGS_DIR + " ".join([now, model, param])
        print("saving learning {}".format(filename), flush=True)
        f = open(filename,'wb')
        pickle.dump(self, f)
        f.close()

    def delete_isolated_states(self):
        """This function deletes the isolated states from the learned model"""
        states = self.model.states
        isolated_states = []
        gamma = self.model.gamma
        for s in range(0, len(states)):
            if np.all(gamma[:,s] == s) and np.all(np.delete(gamma,s,1) != s):
                isolated_states.append(s)
        print('removing the following isolated states {}'.format(isolated_states))
        for i in range(len(isolated_states)):
            s = isolated_states[i] - i
            self.model.nr_of_states -= 1
            self.model.states = range(self.model.nr_of_states)
            self.model.mu = np.delete(self.model.mu,s,0)
            self.model.cov = np.delete(self.model.cov,s,0)
            self.model.gamma = np.delete(self.model.gamma,s,1)
            self.model.gamma -= self.model.gamma > s
            new_T_TABLE = defaultdict(int)
            for sas in self.TRANS:
                new_sas = tuple(np.array(sas) - [sas[0] > s,0,sas[2] > s])
                new_T_TABLE[new_sas] = self.TRANS[sas]
            new_O_TABLE = defaultdict(list)
            for s1 in self.OBS:
                new_O_TABLE[s1 - (s1 > s)] = self.OBS[s1]

def load_learner(filename):
    f = open(filename, "rb")
    print("loading learner {}".format(filename))
    learner = pickle.load(f)
    f.close()
    return learner

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('number_of_trucks', help="specify the number of trucks",type=int)
    parser.add_argument('-rp','--randomposition',help="if it is specified then the block can be placed in a random position on the table",action='store_true')
    parser.add_argument('-i', '--iterations', help="number of iterations",type=int,default=5000)
    parser.add_argument('-iv', '--init_var', help="initial covariance",type=float,default=.5, nargs='+')
    parser.add_argument('-a', '--alpha', help="alpha parameter",type=float,default=.1)
    parser.add_argument('-b', '--beta', help="beta parameter",type=float,default=.99)
    parser.add_argument('-e', '--epsilon', help="epsilon parameter",type=float,default=.7)

    from logistics_simulator import TruckWorld
    args = parser.parse_args()
    nt = args.number_of_trucks
    fp = not args.randomposition
    if nt > 24:
        print('You have specified too many trucks. ALP will consider only 24 trucks')
        nt = 24
    iv = args.init_var

    with open('data.txt') as json_file:
        data = json.load(json_file)

    gps_map = {}
    city_labels = []
    for p in data['cities']:
        city_labels.append(p['id'])
        gps_map[p['id']] = p['gps']

    location_labels = []
    for p in data['locations']:
        location_labels.append(p['id'])
        gps_map[p['id']] = p['gps']

    truck_labels = []
    truck_states = []
    for p in data['trucks']:
        truck_labels.append(p['id'])
        gps_map[p['id']] = p['gps']

    package_labels = []
    for p in data['packages']:
        package_labels.append(p['id'])
        gps_map[p['id']] = p['gps']

    m = Model(TruckWorld(truck_labels, package_labels, location_labels, city_labels, gps_map, noise=0.01), init_cov=iv)

    alpha = args.alpha
    beta = args.beta
    epsilon = args.epsilon

    l = Learner(m, epsilon=epsilon, alpha=alpha, beta=beta)
    nr_of_iterations = args.iterations
    print("-------learning for {} trucks".format(nt))
    print("-------iterations = {}".format(nr_of_iterations))
    print("TIME,    ITERATION,  NR_OF_STATES,   CORRECTNESS,    COMPLETENESS,   REDUNDANCY")
    l.learn(nr_of_iterations, alpha=alpha, beta=beta, epsilon=epsilon, eval_frequency=10, save_frequency=nr_of_iterations+2)
    # m.plot(show=True)



