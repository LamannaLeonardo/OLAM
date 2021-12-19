import sys
import argparse


from Model import *
from Learner import *
from logistics_simulator import TruckWorld

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

ROOT_DIR = "myfiles/Analysis/"
ALL_MODELS_DIR = "{}Models/".format(ROOT_DIR)
ROOT_TEST_DIR = "myfiles/Analysis/Tests/"
LOGS_DIR = "Logs/"
PLOTS_DIR = "Plots/"
MODELS_DIR = "Models/"



def load_model(filename):
    """
    Load an already existent model contained into the "MODELS_DIR" directory. The model file format is ".pickle".
    :param filename: name of the model .pickle file
    :return: the loaded model
    """
    f = open(filename, "rb")
    print("Initial model file: {}".format(filename))
    m = pickle.load(f)
    f.close()
    return m



def create_model_from_scratch(initial_noise, initial_variance):

    with open('data.txt') as json_file:
        data = json.load(json_file)

    gps_map = {}
    city_labels = []
    for p in data['cities']:
        city_labels.append(p['id'])
        gps_map[p['id']] = p['gps']

    location_labels = []
    airport_labels = []
    for p in data['locations']:
        location_labels.append(p['id'])
        gps_map[p['id']] = p['gps']

        if p['airport'] == 1:
            airport_labels.append(p['id'])

    truck_labels = []
    for p in data['trucks']:
        truck_labels.append(p['id'])
        gps_map[p['id']] = p['gps']

    package_labels = []
    for p in data['packages']:
        package_labels.append(p['id'])
        gps_map[p['id']] = p['gps']

    airplane_labels = []
    for p in data['airplanes']:
        airplane_labels.append(p['id'])
        gps_map[p['id']] = p['gps']

    m = Model(TruckWorld(truck_labels, package_labels, location_labels, airport_labels, city_labels, airplane_labels,
                         gps_map, noise=0.01), init_cov=initial_variance, init_noise=initial_noise)
    return m


def load_learner(filename):
    f = open(filename, "rb")
    print("loading learner {}".format(filename))
    learner = pickle.load(f)
    f.close()
    return learner


def learn_instance():

    l = Learner(m, strategy=init_strategy, epsilon=epsilon, alpha=alpha, beta=beta)

    nr_of_iterations = args.iterations
    log_file = open("{}{}_log".format(path_logs, tot_states), "w")

    # DEBUG
    sys.stdout = log_file

    print("-------learning for {} trucks, {} airplanes, {} packages, {} cities and {} locations"
          .format(len(m.simulator.truck_names), len(m.simulator.airplane_names), len(m.simulator.package_names),
                  len(m.simulator.city_names), len(m.simulator.location_names)))
    print("-------iterations = {}".format(nr_of_iterations))
    print("-------exploration strategy = {}".format(init_strategy))
    print("-------alpha = {}".format(alpha))
    print("-------beta = {}".format(beta))
    print("-------epsilon = {}".format(epsilon))
    print("-------initial variance = {}".format(initial_variance))
    print("-------initial noise = {}".format(m.init_noise))
    print("-------goal id = {}".format(l.primary_goal_id))
    print("-------goal perceptions = {}".format(l.primary_goal))

    l.learn(nr_of_iterations, alpha=alpha, beta=beta, epsilon=epsilon, eval_frequency=10, save_frequency=nr_of_iterations+2)

    log_file.close()

    m.save_model("{}_incomplete".format(tot_states), path_model)



if __name__ == "__main__":

    # init_model = "2916states_complete"
    # m = load_model("{}{}".format(ALL_MODELS_DIR, init_model))
    # correctness(m)

    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations', help="number of iterations",type=int,default=50000000)
    parser.add_argument('-iv', '--init_var', help="initial covariance",type=float,default= .5, nargs='+')
    parser.add_argument('-a', '--alpha', help="alpha parameter",type=float,default=.1)
    parser.add_argument('-b', '--beta', help="beta parameter",type=float,default=.99)
    parser.add_argument('-e', '--epsilon', help="epsilon parameter",type=float,default=.0)
    parser.add_argument('-in', '--init_noise', help="initial noise",type=float,default=.5, nargs='+')
    parser.add_argument('-m', '--model', help="initial model .pickle file name", type=str, default=None)
    parser.add_argument('-s', '--strategy', help="initial model strategy (default is random)", type=str, default="random")
    args = parser.parse_args()
    iv = args.init_var
    init_model = args.model
    init_strategy = args.strategy
    alpha = args.alpha
    beta = args.beta
    epsilon = args.epsilon
    initial_variance = args.init_var




    # Initialize environment model from scratch or starting from an already existent one
    if init_model is not None:
        #  Load existing model
        print("Loading already existent model")
        m = load_model("{}{}".format(ALL_MODELS_DIR, init_model))
        assert (m is not None), "Problem loading model"
    else:
        # Create model from scratch
        print("Creating model from scratch")
        m = create_model_from_scratch(args.init_noise, args.init_var)
        assert (m is not None), "Problem creating model from input file \"data.txt\""

    # Create the directory
    dir_counter = 0

    path_root = "{}{}/".format(ROOT_TEST_DIR, "{}_test".format(m.simulator.count_ideal_states()))
    path_logs = "{}{}/{}".format(ROOT_TEST_DIR, "{}_test".format(m.simulator.count_ideal_states()), LOGS_DIR)
    path_model = "{}{}/{}".format(ROOT_TEST_DIR, "{}_test".format(m.simulator.count_ideal_states()), MODELS_DIR)

    while os.path.isdir(path_root):
        dir_counter = dir_counter + 1

        path_root = "{}{}({})/"\
            .format(ROOT_TEST_DIR, "{}_test".format(m.simulator.count_ideal_states()), dir_counter)
        path_logs = "{}{}({})/{}"\
            .format(ROOT_TEST_DIR, "{}_test".format(m.simulator.count_ideal_states()), dir_counter, LOGS_DIR)
        path_model = "{}{}({})/{}"\
            .format(ROOT_TEST_DIR, "{}_test".format(m.simulator.count_ideal_states()), dir_counter, MODELS_DIR)

    while os.path.isdir(path_logs):
        dir_counter = dir_counter + 1

        path_logs = "{}{}({})/{}"\
            .format(ROOT_TEST_DIR, "{}_test".format(m.simulator.count_ideal_states()), dir_counter, LOGS_DIR)
        path_model = "{}{}({})/{}"\
            .format(ROOT_TEST_DIR, "{}_test".format(m.simulator.count_ideal_states()), dir_counter, MODELS_DIR)

    dir_counter = 1

    while os.path.isdir(path_model):
        dir_counter = dir_counter + 1

        path_model = "{}{}({})/{}"\
            .format(ROOT_TEST_DIR, "{}_test".format(m.simulator.count_ideal_states()), dir_counter, MODELS_DIR)

    try:
        os.makedirs(path_root)
        os.makedirs(path_logs)
        os.makedirs(path_model)
    except OSError:
        print("Creation of the directory %s failed" % path_root)
        print("Creation of the directory %s failed" % path_logs)
        print("Creation of the directory %s failed" % path_model)

    # Create the initial knowledge base
    pddl_generator.generate_pddl_facts(m)

    old_stdout = sys.stdout

    tot_states = m.simulator.count_ideal_states()



    learn_instance()
