import argparse
import os.path
import sys

import Configuration
from Util.Simulator import Simulator
from Util import preprocessing, LogReader, Dataframe_generator
from OLAM.Learner import *
from Util.PddlParser import PddlParser

# import gym
# import pddlgym # Do not delete this if you want to use pddlgym

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def get_operator_preconditions(operator):
    with open("PDDL/domain_input.pddl", "r") as f:
    # with open("PDDL/domain_learned.pddl", "r") as f:
        data = [el.strip() for el in f.read().split("\n")]

        all_action_schema = " ".join(data)[" ".join(data).index(":action"):]
        # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
        action_schema = re.findall(":action {}(.*?):effect".format(operator), all_action_schema)[0]
        preconds = sorted(re.findall("\([^()]*\)", action_schema[action_schema.find("precondition"):]))

    return preconds

def compute_all_actionFF():
    """
    Compute all action list through "adl2strips" with pddl problem files
    :return: None
    """

    op_input = list(get_operator_signatures().keys())

    op_not_learned = []
    for op in op_input:
        op_prec = get_operator_preconditions(op)

        op_prec = [el for el in op_prec if el.find("(and )") == -1]

        if len(op_prec) == 0:
            op_not_learned.append(op)

    all_action_op_not_learned = compute_all_action_of_ops(op_not_learned)






    # Copy input domain to a temp one
    shutil.copyfile("PDDL/domain_input.pddl", "PDDL/domain_input_tmp.pddl")

    with open("PDDL/domain_input_tmp.pddl", "r") as f:
        data = f.read().split("\n")

    # Remove not learned operators

    with open("PDDL/domain_input_tmp.pddl", "w") as f:

        removed_rows = []
        for i in range(len(data)):

            if data[i].find(":action") != -1 and data[i].strip().split()[1] in op_not_learned:
                removed_rows.extend(list(range(i, i+5)))

        [f.write(data[i] + "\n") for i in range(len(data)) if i not in removed_rows]


    with open("PDDL/domain_input_tmp.pddl", "r") as f:
        data = f.read().split("\n")


    # Get all possible effects
    with open(os.path.join("PDDL", "operator_uncertain_positive_effects.json")) as f:
        operator_uncertain_positive_effects = json.load(f)

    with open("PDDL/domain_input_tmp.pddl", "w") as f:

        for i in range(len(data)):

            if data[i].find(":predicates") != -1:

                all_obj = get_all_object()

                all_obj_fict_preds = ["(appear_{} ?obj - {})".format(k, k) for k in all_obj.keys()]

                data[i] = data[i] + "\n" + "\n".join(all_obj_fict_preds)

                data[i] = data[i] + "\n(true )"

            elif data[i].find(":action") != -1:

                op_name = data[i].strip().split()[1]
                op_params = [el for i,el in enumerate(data[i+1].replace(":parameters", "").strip()[1:-1].split()) if el.startswith("?")]

                # op_params_types = [el for i,el in enumerate(data[i+1].replace(":parameters", "").strip()[1:-1].split())
                #                    if not el.startswith("?") and el.strip() != "-"]

                single_obj_count = 0
                op_params_types = []
                row = [el for el in data[i+1].replace(":parameters", "").strip()[1:-1].split() if el.strip() != "-"]
                for el in row:
                    if el.startswith("?"):
                        single_obj_count += 1
                    else:
                        [op_params_types.append(el) for _ in range(single_obj_count)]
                        single_obj_count = 0

                op_effect = data[i+5].replace(":effect", "")

                if op_effect.find("(and") != -1:
                    op_effect = op_effect.replace("(and ", "")
                    # op_effect = op_effect.strip()[:-1]
                    op_effect = op_effect.strip()[:-2]

                fictitious_eff = ""

                for param in op_params:

                    if " ".join(data[i+2:i+6]).find(param + ")") == -1 and " ".join(data[i+2:i+6]).find(param + " ") == -1:
                        n = op_params.index(param)
                        fictitious_eff += "(appear_{} ?param_{})".format(op_params_types[n], n+1)

                # fictitious_eff = " ".join(["(appear_{} ?param_{})".format(op_params_types[n], n+1) for n in range(len(op_params_types))])

                data[i + 5] = ":effect (and {}))".format(fictitious_eff + op_effect + " " + " ".join(operator_uncertain_positive_effects[op_name]))


        # Add fictitious action
        for i in range(len(data)):
            if data[i].find("(:action") != -1:
                data[i] = "(:action fict\n:parameters ()\n:precondition(and)\n:effect(true))"+ "\n" + data[i]
                break

        # Write new domain temp file
        [f.write(line + "\n") for line in data]



    # Copy facts file to a temp one and remove goal
    shutil.copyfile("PDDL/facts.pddl", "PDDL/facts_tmp.pddl")

    with open("PDDL/facts_tmp.pddl", "r") as f:
        data = f.read().split("\n")

    with open("PDDL/facts_tmp.pddl", "w") as f:

        for i in range(len(data)):

            if data[i].find(":goal") != -1:

                for j in range(i+1, len(data)):
                    data[j] = ""

                if data[i].strip().startswith(")"):
                    data[i] = ")\n(:goal (and (true))))"
                else:
                    data[i] = "(:goal (and (true))))"

        [f.write(el + "\n") for el in data]



    bash_command = "Planners/FF/ff -o PDDL/domain_input_tmp.pddl -f PDDL/facts_tmp.pddl -i 114  >> outputff.txt"

    process = subprocess.Popen(bash_command, shell=True)
    process.wait()
    # print("(Preprocessing) -- ADL2STRIPS Finished!")
    #
    # print("(Preprocessing) -- Reading ADL2STRIPS output...")

    action_labels = []
    with open("outputff.txt", "r") as ground_actions_file:
        data = ground_actions_file.read().split("\n")

        for i in range(len(data)):
            line = data[i]

            if line.find("-----------operator") != -1:
                op_name = line.split()[1].split(":")[0].strip().lower()

                if op_name.strip() != "fict":

                    for j in range(i+1, len(data)):
                        if data[j].find("-----------operator") != -1 or data[j].find("Cueing down from goal distance") != -1:
                            break

                        action_obj = [el.lower() for k,el in enumerate(data[j].replace(",", "").split()) if k%3==0][1:]

                        if len(action_obj) > 0:
                            action_labels.append("{}({})".format(op_name, ",".join(action_obj)))

    # print("(Preprocessing) -- Reading ADL2STRIPS finished!")

    action_labels = sorted(action_labels)

    # Remove FF files
    os.remove("PDDL/domain_input_tmp.pddl")
    os.remove("PDDL/facts_tmp.pddl")
    os.remove("outputff.txt")

    return sorted(action_labels + all_action_op_not_learned)



def compute_all_actionADL():
    """
    Compute all action list through "adl2strips" with pddl problem files
    :return: None
    """

    # print("(Preprocessing) -- Calling ADL2STRIPS to get input action list...")

    # bash_command = "Planners/ADL2STRIPS/adl2strips -o PDDL/domain_learned.pddl -f PDDL/facts.pddl"

    # bash_command = "Planners/ADL2STRIPS/adl2strips -o PDDL/domain.pddl -f PDDL/facts.pddl"

    bash_command = "Planners/ADL2STRIPS/adl2strips -o PDDL/domain_input.pddl -f PDDL/facts.pddl"

    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    process.wait()
    # print("(Preprocessing) -- ADL2STRIPS Finished!")
    #
    # print("(Preprocessing) -- Reading ADL2STRIPS output...")

    with open(Configuration.ADL2STRIPS_FILE, "r") as ground_actions_file:
        data = ground_actions_file.read().split("\n")

        action_labels = [row[8:-2].strip().lower().replace("- ", "(", 1).replace("- ",",") + ")"
                         for row in filter(lambda k: '(:action' in k, data)]

    # print("(Preprocessing) -- Reading ADL2STRIPS finished!")

    # Remove ADL2STRIPS files
    os.remove(Configuration.ADL2STRIPS_FILE)
    os.remove("facts.pddl")

    return action_labels



def compute_all_action():
    """
    Compute all action list through cartesian product of input objects
    :return: None
    """

    all_action_labels = []

    all_objs = get_all_object()

    all_op = get_operator_signatures()

    obj_types = get_object_types_hierarchy()

    for op in all_op.keys():

        # Compute all combinations of action input object types, subclassing all supertypes
        subclass_obj_types = [obj_types[el] if len(obj_types[el]) > 0 else [el] for el in all_op[op]]
        subclass_obj_types = [list(p) for p in itertools.product(*subclass_obj_types)]

        for tuple_input_obj in subclass_obj_types:
            op_obj_lists = [all_objs[obj_key] for obj_key in tuple_input_obj]

            all_obj_combinations = itertools.product(*op_obj_lists)

            [all_action_labels.append("{}({})".format(op, ",".join(objs))) for objs in all_obj_combinations]

    return all_action_labels



def compute_all_action_of_ops(operators):
    """
    Compute all action list through cartesian product of input objects
    :return: None
    """

    all_action_labels = []

    all_objs = get_all_object()

    all_op = get_operator_signatures()

    obj_types = get_object_types_hierarchy()

    for op in [el for el in all_op.keys() if el in operators]:

        # Compute all combinations of action input object types, subclassing all supertypes
        subclass_obj_types = [obj_types[el] if len(obj_types[el]) > 0 else [el] for el in all_op[op]]
        subclass_obj_types = [list(p) for p in itertools.product(*subclass_obj_types)]

        for tuple_input_obj in subclass_obj_types:
            op_obj_lists = [all_objs[obj_key] for obj_key in tuple_input_obj]

            all_obj_combinations = itertools.product(*op_obj_lists)

            [all_action_labels.append("{}({})".format(op, ",".join(objs))) for objs in all_obj_combinations]

    return all_action_labels


def get_operator_signatures():
    all_op_names = []

    with open("PDDL/domain_learned.pddl", "r") as f:

        data = [el.strip() for el in f.read().split("\n")]

        for line in data:

            if line.strip().find("(:action") != -1:
                all_op_names.append(line.split()[1].strip())

    all_op_objs = defaultdict(list)

    with open("PDDL/domain_learned.pddl", "r") as f:
        data = [el.strip() for el in f.read().split("\n")]

        for operator in all_op_names:
            all_action_schema = " ".join(data)[" ".join(data).index(":action"):]
            # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
            action_schema = re.findall(":action {}(.*?):effect".format(operator), all_action_schema)[0]
            op_params = re.findall(":parameters(.*?):precondition", action_schema)[0].strip()

            single_obj_count = 0
            for el in [el for el in op_params.strip()[1:-1].split() if el.strip() != "-"]:
                if el.startswith("?"):
                    single_obj_count += 1
                else:
                    [all_op_objs[operator].append(el) for _ in range(single_obj_count)]
                    single_obj_count = 0

            # all_op_objs[operator] = [el.split()[0] for el in op_params.strip()[:-1].split("- ")[1:]]

    return all_op_objs

def get_all_object():

    with open("PDDL/facts.pddl", "r") as f:
        data = [el.strip() for el in f.read().split("\n") if not el.strip().startswith(";")]
        obj_list = re.findall(":objects.*:init", "++".join(data))[0].replace(":objects", "").replace("(:init", "")

    obj_list = [el.replace(")","") for el in obj_list.split("++") if el.strip() != "" and el.strip() != ")"]

    new_obj_list = []

    for el in obj_list:

        if el.find(" -") != -1:
            while el.count(" -") > 0:
                new_obj_list.append(el.split(" -")[0] + " - " + el.split(" -")[1].split()[0])
                el = " ".join(" - ".join(el.split(" -")[1:]).split()[1:])
        else:
            new_obj_list.append(el)

    obj_list = new_obj_list

    all_objs = defaultdict(list)

    obj_of_type_list = []
    for line in obj_list:

        if line.find("- ") != -1:

            obj_type = line[::-1].split(" -")[0][::-1].strip().lower()

            obj_of_type_list.extend(line.lower().replace("- " + obj_type, "").strip().split())
            # all_objs[obj_type] = obj_of_type_list
            all_objs[obj_type].extend(obj_of_type_list)
            obj_of_type_list = []
        else:
            obj_of_type_list.extend(line.split())

    return all_objs


def get_object_types_hierarchy():
    with open("PDDL/domain_learned.pddl", 'r') as f:
        data = f.read().split("\n")

        objects_row = [el.replace(")","").strip()
                       for el in re.findall(":types.*\(:predicates","++".join(data))[0].replace(":types","").replace("(:predicates", "").split("++")
                       if el.strip() != ""]

        objects = defaultdict(list)

        start_index = 0
        obj_of_same_type = []
        for row in objects_row:
            current_index = len(objects['objects'])
            row = row.replace("(", "").replace(")", "")
            if row.find("- ") != -1:
                [objects['objects'].append(el) for el in row.strip().split("- ")[0].split()]
                [objects['objects'].append(el) for el in row.strip().split("- ")[1].split()]
                # objects[row.strip().split("- ")[1].strip()] = [el.strip()
                #                                               for el in row.strip().split("- ")[0].strip().split()]
                objects[row.strip().split("- ")[1].strip()].extend([el.strip()
                                                                    for el in row.strip().split("- ")[0].strip().split()]
                                                                   + obj_of_same_type
                                                                   + [row.strip().split("- ")[1].strip()])
                start_index = current_index + 1
                obj_of_same_type = []
            else:
                [objects['objects'].append(el) for el in row.split()]
                [obj_of_same_type.append(el) for el in row.split()]

        for object_key, object_values in objects.items():
            if object_key != 'objects':

                for val in object_values:

                    for key in objects.keys():
                        if val == key:
                            # objects[object_key] = [el for el in objects[object_key] + objects[val] if el != val]
                            objects[object_key] = [el for el in objects[object_key] + objects[val]]

        # objects['objects'] = list(set(objects['objects']))

        for key in objects.keys():
            objects[key] = list(set(objects[key]))

    return objects


def learn_instance(path_logs, simulator, parser, all_actions):
    """
    Create the learner, print some starting information, solve the problem instance, store the learnt action model
    and evaluate metrics (e.g. precision, recall, ecc...)
    :param path_logs: log file path
    :param simulator: pddlgym simulator
    :param parser: pddl domain parser
    :param all_actions: list of all domain actions
    :return: None
    """

    # Instantiate the Learner
    l = Learner(parser=parser, action_list=all_actions)

    log_file_path = "{}/{}_log".format(path_logs, Configuration.INSTANCE_DATA_PATH_PDDL.split("/")[-1].split(".")[0])
    log_file = open(log_file_path, "w")

    print("Running OLAM...")

    # print("\nTotal actions: {}".format(len(all_actions)))
    #
    # print("\nObjects list\n\t{}\n\n".format("\n\t".join(["{}:{}".format(k, len(v)) for k,v in get_all_object().items()])))

    old_stdout = sys.stdout

    if not Configuration.OUTPUT_CONSOLE:
        print(f'Standard output redirected to {log_file_path}')
        sys.stdout = log_file

    print("\nTotal actions: {}".format(len(all_actions)))

    print("\nObjects list\n\t{}\n\n".format("\n\t".join(["{}:{}".format(k, len(v)) for k,v in get_all_object().items()])))

    # Learn action model from problem instance
    l.learn(eval_frequency=10, simulator=simulator)

    log_file.close()

    if not Configuration.OUTPUT_CONSOLE:
        LogReader.evaluate_log_metrics(log_file_path)

    sys.stdout = old_stdout

    print("End of OLAM resolution.")


    # Compute learned domain with certain preconditions
    shutil.copyfile("PDDL/domain_learned.pddl", "PDDL/domain_learned_certain.pddl")

    with open("PDDL/domain_learned_certain.pddl", "r") as f:
        data = f.read().split("\n")

    with open("PDDL/domain_learned_certain.pddl", "w") as f:

        for i in range(len(data)):

            line = data[i]

            if line.find(":action") != -1:
                op_name = line.split()[1]

                precond = sorted(re.findall("\([^()]*\)", data[i+3]))

                to_remove = []
                for prec in precond:
                    if prec not in l.operator_certain_predicates[op_name]:
                        to_remove.append(prec)

                if len([prec for prec in precond if prec not in to_remove]) > 0:
                    data[i+3] = "\t\t"+ " ".join([prec for prec in precond if prec not in to_remove])
                else:
                    data[i+3] = ")"

        [f.write(line + "\n") for line in data]

    # Save uncertain preconditions of each learned operator
    with open(os.path.join("PDDL", "operator_uncertain_precs.json"), "w") as outfile:
        # json.dump(self.operator_negative_preconditions, outfile)
        json.dump(l.operator_uncertain_predicates, outfile, indent=2)

    shutil.copyfile(os.path.join("PDDL", "operator_uncertain_precs.json"),
                    os.path.join(path_logs, "operator_uncertain_precs.json"))

    # Save certain positive effects of each learned operator
    with open(os.path.join("PDDL", "operator_certain_positive_effects.json"), "w") as outfile:
        # json.dump(self.operator_negative_preconditions, outfile)
        json.dump(l.certain_positive_effects, outfile, indent=2)

    shutil.copyfile(os.path.join("PDDL", "operator_certain_positive_effects.json"),
                    os.path.join(path_logs, "operator_certain_positive_effects.json"))

    # Save certain negative effects of each learned operator
    with open(os.path.join("PDDL", "operator_certain_negative_effects.json"), "w") as outfile:
        # json.dump(self.operator_negative_preconditions, outfile)
        json.dump(l.certain_negative_effects, outfile, indent=2)

    shutil.copyfile(os.path.join("PDDL", "operator_certain_negative_effects.json"),
                    os.path.join(path_logs, "operator_certain_negative_effects.json"))

    # Save potentially possible positive effects of each learned operator,
    # i.e., effects that may be learned in a different problem
    with open(os.path.join("PDDL", "operator_uncertain_positive_effects.json"), "w") as outfile:
        # json.dump(self.operator_negative_preconditions, outfile)
        json.dump(l.uncertain_positive_effects, outfile, indent=2)

    shutil.copyfile(os.path.join("PDDL", "operator_uncertain_positive_effects.json"),
                    os.path.join(path_logs, "operator_uncertain_positive_effects.json"))

    # Save potentially possible negative effects of each learned operator,
    # i.e., effects that may be learned in a different problem
    with open(os.path.join("PDDL", "operator_uncertain_negative_effects.json"), "w") as outfile:
        # json.dump(self.operator_negative_preconditions, outfile)
        json.dump(l.uncertain_negative_effects, outfile, indent=2)

    shutil.copyfile(os.path.join("PDDL", "operator_uncertain_negative_effects.json"),
                    os.path.join(path_logs, "operator_uncertain_negative_effects.json"))

    # Save useless possible preconditions of not yet learned operators,
    # i.e., possible preconditions which has been satisfied during a previous resolution but for which
    # the action has not been executable
    with open(os.path.join("PDDL", "operator_useless_possible_precs.json"), "w") as outfile:
        # json.dump(self.operator_negative_preconditions, outfile)
        json.dump(l.useless_possible_precs, outfile, indent=2)

    shutil.copyfile(os.path.join("PDDL", "operator_useless_possible_precs.json"),
                    os.path.join(path_logs, "operator_useless_possible_precs.json"))

    # Save useless negated preconditions of not learned operators,
    # i.e., preconditions that has been negated during a previous resolution but for which
    # the action has not been executable
    with open(os.path.join("PDDL", "operator_useless_negated_precs.json"), "w") as outfile:
        # json.dump(self.operator_negative_preconditions, outfile)
        json.dump(l.useless_negated_precs, outfile, indent=2)

    shutil.copyfile(os.path.join("PDDL", "operator_useless_negated_precs.json"),
                    os.path.join(path_logs, "operator_useless_negated_precs.json"))


def solve_instance():
    """
    Solve problem instance applying the following steps: Create the domain simulator,
    create problem instance log directories and solve problem instance
    :return: None
    """

    # Create the simulator
    simulator = Simulator()

    # Get all actions list (this should be an input, or alternatively a superset of all possible actions which
    # could be automatically computed by the learner)
    # all_actions = compute_all_action()

    op_input = list(get_operator_signatures().keys())
    op_not_learned = []

    if os.path.exists("PDDL/domain_input.pddl"):

        for op in op_input:
            op_prec = get_operator_preconditions(op)

            op_prec = [el for el in op_prec if el.find("(and )") == -1]

            if len(op_prec) == 0:
                op_not_learned.append(op)

    if os.path.exists("PDDL/domain_input.pddl") and len(op_not_learned) == 0:
        # all_actions = compute_all_actionADL()
        all_actions = compute_all_actionFF()

        if len(all_actions) == 0:
            print('Warning: bug in FF when computing all actions, using cartesian product')
            all_actions = compute_all_action()
    else:
        all_actions = compute_all_action()

    # Create the instance logs directory
    dir_counter = 0
    # path_root = "{}{}/{}/{}/".format(Configuration.ROOT_TEST_DIR, domain, Configuration.BENCHMARK_DIR,
    #                                  instance_name.split('.')[0])
    path_root = os.path.join(Configuration.ROOT_TEST_DIR, domain, Configuration.BENCHMARK_DIR,
                             instance_name.split('.')[0])

    while os.path.isdir(path_root):
        dir_counter = dir_counter + 1
        # path_root = "{}{}/{}/{}({})".format(Configuration.ROOT_TEST_DIR, domain, Configuration.BENCHMARK_DIR,
        #                                      instance_name.split('.')[0], dir_counter)
        path_root = os.path.join(Configuration.ROOT_TEST_DIR, domain, Configuration.BENCHMARK_DIR,
                                 f"{instance_name.split('.')[0]}({dir_counter})")

    try:
        os.makedirs(path_root)
    except OSError:
        print("Creation of the directory %s is failed" % path_root)

    # Instantiate PDDL parser and update initial PDDL state
    parser = PddlParser()
    # parser.update_pddl_facts(obs)

    # Solve problem instance
    learn_instance(path_root, simulator, parser, all_actions)

    # Save learned domain
    shutil.copyfile("PDDL/domain_learned.pddl", os.path.join(path_root, "domain_learned.pddl"))

    # Save learned domain with certain preconditions
    shutil.copyfile("PDDL/domain_learned_certain.pddl", os.path.join(path_root, "domain_learned_certain.pddl"))

    # Save input domain of solved problem, if it exists
    if os.path.exists("PDDL/domain_input.pddl"):
        shutil.copyfile("PDDL/domain_input.pddl", os.path.join(path_root, "domain_input.pddl"))

    # Save learned domain as input domain for the next problem
    shutil.copyfile("PDDL/domain_learned_certain.pddl", "PDDL/domain_input.pddl")

    # Save reached state
    shutil.copyfile("PDDL/facts.pddl", os.path.join(path_root, "final_state.pddl"))


if __name__ == "__main__":

    # Set input arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('-d', '--domain', help="Domain name (must be equal to domain benchmark instances root directory)",
                        type=str, default=None)

    # Get input arguments
    args = args_parser.parse_args()
    domain = args.domain

    # Check input arguments
    assert (Configuration.MAX_ITER > 0), "MAX_ITER in Configuration.py must be greater than 0"
    assert (isinstance(Configuration.NEG_EFF_ASSUMPTION, bool)), "NEG_EFF_ASSUMPTION in Configuration.py must be True or " \
                                                                 "False, default value is False"
    assert (isinstance(domain, str) or domain is None), "-domain must be a string equal to a domain benchmark instances root directory"
    assert (domain in os.listdir(os.path.join("Analysis", "Benchmarks"))
            or domain is None), "-domain must be equal to a domain benchmark " \
                                "instances root directory (in Analysis/Benchmarks)"

    java_jdk_dir = [d for d in os.listdir(os.path.join(os.getcwd(), Configuration.JAVA_DIR))
                    if os.path.isdir(os.path.join(os.getcwd(), Configuration.JAVA_DIR, d))]

    if len(java_jdk_dir) == 0:
        print('\n\nMissing oracle jdk directory in "Java" directory. Please download oracle jdk tarball and extract it '
              'into "Java" directory.')

    elif len(java_jdk_dir) > 1:
        print(f'\n\nMultiple jdk directories in "Java" directory. Please delete all jdk directories in "Java" '
              f'directory but the chosen one. I am trying to execute the program by looking for java binary '
              f'in {os.path.join(os.getcwd(), Configuration.JAVA_DIR, java_jdk_dir[0])}.')

    java_jdk_dir = java_jdk_dir[0]
    Configuration.JAVA_BIN_PATH = os.path.join(os.getcwd(), Configuration.JAVA_DIR, java_jdk_dir, "bin", "java")

    assert os.path.exists(Configuration.JAVA_BIN_PATH), f"File not found: {Configuration.JAVA_BIN_PATH}"

    assert (isinstance(Configuration.OUTPUT_CONSOLE, bool)), "OUTPUT_CONSOLE in Configuration.py must be True or False"


    all_domains = []
    if domain is None:
        all_domains = [el for el in os.listdir(os.path.join("Analysis", "Benchmarks"))
                       if not el.endswith(".pddl")]
        print('\n\nRunning OLAM over all domain in Analysis/Benchmarks directory')
    else:
        all_domains = [domain]
        print(f'\n\nRunning OLAM in {domain} domain\n')


    # Set test directory
    runs = [d for d in os.listdir(Configuration.ROOT_DIR) if d.startswith('run_')]
    Configuration.ROOT_TEST_DIR = os.path.join(Configuration.ROOT_DIR, f"run_{len(runs)}", "Tests")
    # Configuration.ROOT_TEST_DIR = "{}Tests/".format(Configuration.ROOT_DIR)

    for domain in all_domains:
        # Domain benchmarks directory
        instances_dir = "{}{}".format(Configuration.ROOT_BENCHMARKS_DIR, domain)

        # Clean working files in PDDL directory
        clean = False
        if os.path.exists("PDDL/domain_input.pddl"):

            with open("PDDL/domain_input.pddl", "r") as f:

                for el in f.read().split("\n"):

                    if el.find("(domain") != -1:

                        # Special case for nomystery
                        if domain == "nomystery" and "transport" in el.lower().strip().split()[2].replace("-", ""):
                            clean = False
                            break

                        if domain.lower().replace("-", "") not in el.lower().strip().split()[2].replace("-", ""):
                            clean = True
                            break

                        else:
                            break

        if clean:
            shutil.rmtree("PDDL")
            os.mkdir("PDDL")

        all_instances = None
        try:
            all_instances = sorted(os.listdir(instances_dir), key=lambda x: int(x.split("_")[0]))
        except ValueError:
            print("All instance file names in domain benchmark directory {} must begin with "
                  "a number followed by underscore, e.g. 1_instancename".format(instances_dir))

        assert all_instances is not None, print("All instance file names in domain benchmark directory {} must begin with "
                                                "a number followed by underscore, e.g. 1_instancename. Moreover, the domain "
                                                "benchmark directory must be into \"Analysis/Benchmarks\" directory".format(instances_dir))


        for instance_name in all_instances:

            # Set instance file name and path
            Configuration.INSTANCE_DATA_PATH_PDDL = os.path.join("Analysis", "Benchmarks", domain, instance_name)

            # Copy original domain and problem instance to working files
            preprocessing.preprocess(domain)

            # Clean temporary files (i.e., not executable actions files)
            if os.path.exists("Info"):
                shutil.rmtree("Info")
            os.mkdir("Info")

            # print("\n\n +-+-+-+-+-+-+-+-+-+-+-+-+-+ OLAM +-+-+-+-+-+-+-+-+-+-+-+-+-+\n")
            print(f"\nSolving instance {Configuration.INSTANCE_DATA_PATH_PDDL}")

            if os.path.exists("PDDL/domain_input.pddl"):
                print("Reading input domain PDDL/domain_input.pddl, if you do not want to use an input domain, make "
                      "the PDDL directory empty")

            # Solve instance
            solve_instance()

        # Clean not executable action files and PDDL files
        shutil.rmtree("Info")
        shutil.rmtree("PDDL")

        if not Configuration.OUTPUT_CONSOLE:
            # Generate final results without uncertain negative effects
            if not Configuration.NEG_EFF_ASSUMPTION:
                Dataframe_generator.generate_domain_dataframes()
                Dataframe_generator.generate_domain_summary()

            # Generate final results with uncertain negative effects
            uncert_neg_effects = True
            Dataframe_generator.generate_domain_dataframes(uncert_neg_effects)
            Dataframe_generator.generate_domain_summary(uncert_neg_effects)