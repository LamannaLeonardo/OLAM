# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import json
import Configuration
import itertools
from timeit import default_timer
from collections import defaultdict

from OLAM import Planner
from Util import metrics

template = "{Time:^8}|{Iter:^6}|{Real_precs:^10}|{Learn_precs:^11}|{Real_pos:^8}|{Learn_pos:^9}" \
           "|{Real_neg:^8}|{Learn_neg:^9}|{Ins_pre:^7}|{Del_pre:^7}|{Ins_pos:^7}|{Del_pos:^7}" \
           "|{Ins_neg:^7}|{Del_neg:^7}|{Precs_recall:^12}|{Pos_recall:^10}|{Neg_recall:^10}" \
           "|{Precs_precision:^15}|{Pos_precision:^13}|{Neg_precision:^13}" \
           "|{Tot_recall:^10}|{Tot_precision:^13}"

class Learner:
    def __init__(self, parser, action_list, eval_frequency=10):

        # Set random seed
        np.random.seed(Configuration.RANDOM_SEED)

        # PDDL parser, used to update pddl problem file from pddlgym simulator state
        self.parser = parser

        # Actions labels
        self.action_labels = sorted([el.lower() for el in action_list])

        # Operator static and dynamic preconditions
        self.operator_params = self.get_operator_params()

        self.all_operators = self.operator_params.keys()

        # Uncertain learnt preconditions
        self.operator_uncertain_predicates = defaultdict(list)

        # Certain learnt preconditions
        self.operator_certain_predicates = defaultdict(list)
        # self.operator_certain_predicates = self.init_certain_predicates()
        self.operator_learned = []
        # self.operator_learned = self.init_op_learned()

        # Constraint on action executability
        # self.operator_executability_constr = defaultdict(list)
        self.operator_executability_constr = self.init_operator_exec_constr()
        self.operator_negative_preconditions = defaultdict(list)

        # These are fictitious preconditions exploited by adl2strips to compute a smaller list of not executable actions

        # True if the action model preconditions have been updated since last plan computation
        self.changed_preconditions = False

        self.action_precondition_percentage = [float(1) for _ in self.action_labels]
        self.not_executable_actions = []
        self.executable_actions = []
        self.action_precond_perc_filtered = [self.action_precondition_percentage[i]
                                             # if i not in executable_actions + self.tried_actions else -1
                                             for i in range(len(self.action_precondition_percentage))]

        self.checking_precondition = False
        self.checked_precondition = None

        self.tried_actions = []

        self.not_executable_actions_index = []

        self.initial_timer = None
        self.max_time_limit = None

        self.last_failed_action = None
        self.last_action = None

        # Current plan
        self.current_plan = None

        self.eval_frequency = eval_frequency
        self.eval = pd.DataFrame(columns=('timestamp','iter','nr_of_states','recall','precision'))

        self.iter = 0
        self.time_at_iter = [0.]

        self.model_convergence = False

        # self.uncertain_positive_effects = defaultdict(list)
        # self.uncertain_negative_effects = defaultdict(list)

        # self.uncertain_positive_effects = {op:self.get_op_relevant_predicates(op) for op in self.all_operators}
        self.uncertain_positive_effects = self.init_op_uncertain_positive_effects()
        # self.uncertain_negative_effects = {op:self.get_op_relevant_predicates(op) for op in self.all_operators}
        self.uncertain_negative_effects = self.init_op_uncertain_negative_effects()

        # self.certain_positive_effects = defaultdict(list)
        self.certain_positive_effects = self.init_op_certain_positive_effects()
        # self.certain_negative_effects = defaultdict(list)
        self.certain_negative_effects = self.init_op_certain_negative_effects()

        # List of combinations already tested and satisfied, but for which the action
        # has failed => do not try to satisfy them anymore
        self.useless_possible_precs = self.init_op_useless_possible_precs()
        self.useless_negated_precs = self.init_op_useless_negated_precs()


        # Read input certain and uncertain operator preconditions
        if os.path.exists("PDDL/domain_input.pddl"):

            with open(os.path.join("PDDL", "operator_uncertain_precs.json")) as f:
                self.operator_uncertain_predicates = json.load(f)
                self.operator_uncertain_predicates = defaultdict(list, self.operator_uncertain_predicates)

                for op in list(self.operator_uncertain_predicates.keys()):
                    if len(self.operator_uncertain_predicates[op]) > 0:
                        self.operator_learned.append(op)

            with open("PDDL/domain_input.pddl", "r") as f:
                data = [el.strip() for el in f.read().split("\n")]

                all_action_schema = " ".join(data)[" ".join(data).index(":action"):]
                all_action_schema = re.findall(":action (.*?):effect", all_action_schema)

                for action_schema in all_action_schema:
                    preconds = sorted(re.findall("\([^()]*\)", action_schema[action_schema.find("precondition"):]))

                    preconds = [p for p in preconds if p.strip() != "(and )"]

                    if len(preconds) > 0:
                        op_name = action_schema.strip().split()[0]
                        if op_name not in self.operator_learned:
                            self.operator_learned.append(op_name)
                        self.operator_certain_predicates[op_name] = preconds

        # Update learned domain according to input operator preconditions (both certain and uncertain)
        with open("PDDL/domain_learned.pddl", "r") as f:
            data = f.read().split("\n")

        with open("PDDL/domain_learned.pddl", "w") as f:

            for i in range(len(data)):

                if data[i].find(":action") != -1:
                    op = data[i].strip().split()[1]

                    if op in self.operator_learned:
                        data[i+2] = data[i+2] + "\n {}".format(" ".join(self.operator_certain_predicates[op]
                                                               + self.operator_uncertain_predicates[op]))

            [f.write(row + "\n") for row in data]

            # f.write("\n)")



        # Update domain dummy (i.e., domain with only learned operators)
        shutil.copyfile("PDDL/domain_learned.pddl", os.path.join("PDDL/domain_dummy.pddl"))
        with open("PDDL/domain_dummy.pddl", 'r') as f:
            data = [el for el in f.read().split("\n") if el.strip() != ""]

            removed_rows = []

            for m in range(len(data)):

                row = data[m]

                if row.lower().strip().startswith("(:action"):

                    if data[m + 2].strip().startswith(":precondition") and data[m + 3].strip() == ")":
                        removed_rows.append(m)

                        for n in range(m + 1, len(data) - 1):
                            if data[n].strip().startswith("(:action"):
                                break
                            removed_rows.append(n)

            dummy_domain = [data[i] for i in range(len(data)) if i not in removed_rows]

        with open("PDDL/domain_dummy.pddl", 'w') as f:
            [f.write(el + "\n") for el in dummy_domain]


    def init_op_useless_possible_precs(self):

        if not os.path.exists("PDDL/domain_input.pddl"):
            return defaultdict(list)

        with open(os.path.join("PDDL", "operator_useless_possible_precs.json")) as f:
            useless_possible_precs = json.load(f)

        return defaultdict(list, useless_possible_precs)


    def init_op_useless_negated_precs(self):

        if not os.path.exists("PDDL/domain_input.pddl"):
            return defaultdict(list)

        with open(os.path.join("PDDL", "operator_useless_negated_precs.json")) as f:
            useless_negated_precs = json.load(f)

        return defaultdict(list, useless_negated_precs)


    def init_op_certain_positive_effects(self):

        if not os.path.exists("PDDL/domain_input.pddl"):
            return defaultdict(list)

        with open(os.path.join("PDDL", "operator_certain_positive_effects.json")) as f:
            operator_certain_positive_effects = json.load(f)

        return defaultdict(list, operator_certain_positive_effects)


    def init_op_certain_negative_effects(self):

        if not os.path.exists("PDDL/domain_input.pddl"):
            return defaultdict(list)

        with open(os.path.join("PDDL", "operator_certain_negative_effects.json")) as f:
            operator_certain_negative_effects = json.load(f)

        return defaultdict(list, operator_certain_negative_effects)


    def init_op_uncertain_positive_effects(self):

        if not os.path.exists("PDDL/domain_input.pddl"):
            return {op:self.get_op_relevant_predicates(op) for op in self.all_operators}

        with open(os.path.join("PDDL", "operator_uncertain_positive_effects.json")) as f:
            operator_possible_positive_effects = json.load(f)

        return defaultdict(list, operator_possible_positive_effects)


    def init_op_uncertain_negative_effects(self):

        if not os.path.exists("PDDL/domain_input.pddl"):
            return {op:self.get_op_relevant_predicates(op) for op in self.all_operators}

        with open(os.path.join("PDDL", "operator_uncertain_negative_effects.json")) as f:
            operator_possible_negative_effects = json.load(f)

        return defaultdict(list, operator_possible_negative_effects)


    def init_operator_exec_constr(self):
        if not os.path.exists("PDDL/domain_input.pddl"):
            return defaultdict(list)

        op_exec_constraints = defaultdict(list)
        with open("PDDL/domain_input.pddl", "r") as f:
            data = f.read().split("\n")

        for i in range(len(data)):

            line = data[i]

            if line.find(":action") != -1:
                op_name = line.split()[1]

                precond = sorted(re.findall("\([^()]*\)", data[i + 3]))

                for prec in precond:
                    op_exec_constraints[op_name].append(["not({})".format(prec)])


        return defaultdict(list, op_exec_constraints)


    def init_op_learned(self):
        if not os.path.exists("PDDL/domain_input.pddl"):
            return []

        op_learned = []
        with open("PDDL/domain_input.pddl", "r") as f:
            data = f.read().split("\n")

        for i in range(len(data)):

            line = data[i]

            if line.find(":action") != -1:
                op_name = line.split()[1]

                precond = sorted(re.findall("\([^()]*\)", data[i + 3]))

                if len(precond) > 0:
                    op_learned.append(op_name)


        return op_learned


    def init_certain_predicates(self):
        if not os.path.exists("PDDL/domain_input.pddl"):
            return defaultdict(list)

        operator_certain_predicates = defaultdict(list)
        with open("PDDL/domain_input.pddl", "r") as f:
            data = f.read().split("\n")

        for i in range(len(data)):

            line = data[i]

            if line.find(":action") != -1:
                op_name = line.split()[1]

                precond = sorted(re.findall("\([^()]*\)", data[i + 3]))

                operator_certain_predicates[op_name] = [prec for prec in precond]


        return defaultdict(list, operator_certain_predicates)


    def get_operator_params(self):
        operator_params = defaultdict(list)

        with open("PDDL/domain_learned.pddl", "r") as f:
            data = [el for el in f.read().split("\n") if el.strip() != ""]

        for i in range(len(data)):
            row = data[i]

            if row.strip().startswith(":parameters"):
                operator_params[data[i-1].strip().split()[1]] = [el.strip()[:-1].strip() for el in re.findall("\?[^?]* -",row)]

        return operator_params


    def compute_executable_actions(self):

        # self.action_precondition_percentage = [float(1) for _ in self.action_labels]

        with open("PDDL/facts.pddl", "r") as f:
            data = [el.strip() for el in f.read().split("\n")]
            facts = re.findall(":init.*\(:goal","".join(data))[0]

        current_state_pddl = sorted(re.findall("\([^()]*\)", facts))

        executable_actions = []

        for action_id in range(len(self.action_labels)):

            action_label = self.action_labels[action_id]

            operator = action_label.split("(")[0]

            params = [el.strip() for el in action_label.split("(")[1][:-1].split(",")]

            all_precond = []

            for pred in self.operator_uncertain_predicates[operator] + self.operator_certain_predicates[operator]:

                for k in range(len(params)):
                    pred = pred.replace("?param_{})".format(k + 1), params[k] + ")")
                    pred = pred.replace("?param_{} ".format(k + 1), params[k] + " ")
                all_precond.append(pred)

            # Satisfied preconditions
            sat_precond = sum([True for pred in all_precond if pred in current_state_pddl])

            if len(all_precond) == sat_precond and len(all_precond) > 0:
                executable_actions.append(action_label)

            if len(all_precond) > 0:
                self.action_precondition_percentage[action_id] = sat_precond / len(all_precond)
            else:
                self.action_precondition_percentage[action_id] = 1

        return executable_actions


    def select_action(self):
        """
        Select an action: firstly look if a plan can be computed, otherwise return the action which maximizes the
        satisfied preconditions rate.
        :return: an action name
        """

        # Try planner action, otherwise random action
        if default_timer() - self.initial_timer < self.max_time_limit:

            # Compute not executable actions
            self.not_executable_actions_index = self.compute_not_executable_actionsJAVA()
            self.executable_actions = self.compute_executable_actions()

            # DEBUG
            print("Not executable actions: {}".format(len(self.not_executable_actions_index)))

            self.action_precond_perc_filtered = [el for el in self.action_precondition_percentage]

            for el in self.not_executable_actions_index:
                self.action_precond_perc_filtered[el] = -1

            for el in [self.action_labels.index(a) for a in self.executable_actions]:
                self.action_precond_perc_filtered[el] = -1

            likely_executable_actions = [i for i in
                                          np.argwhere(self.action_precond_perc_filtered == np.amax(self.action_precond_perc_filtered))\
                                          .flatten().tolist()]

            # if len(self.executable_actions) + len(self.not_executable_actions_index) != len(self.action_labels):
            if len(set([self.action_labels.index(a) for a in self.executable_actions]
                       + self.not_executable_actions_index)) != len(self.action_labels):
                print('Random action chosen')
                random_action = np.random.choice(likely_executable_actions)
                return random_action, Configuration.STRATEGIES[1]

            # If there are no new potentially executable actions to be tried, try to satisfy false preconditions
            # of not learned operators
            else:
                print('Random action chosen')
                not_learned_actions = list(filter(lambda action_label:
                                                  action_label.split("(")[0] not in self.operator_learned,
                                                  self.action_labels))

                if len(not_learned_actions) > 0:
                    # print('Random action chosen to satisfy false preconditions of unknown operator')
                    random_action = np.random.choice(not_learned_actions)
                    random_action = self.action_labels.index(random_action)
                    return random_action, Configuration.STRATEGIES[1]
                else:
                    return None, None



    # def get_op_relevant_predicatesNORMAL(self, op_name):
    def get_op_relevant_predicates(self, op_name):

        op_params = self.get_operator_parameters(op_name).strip()[1:-1]

        obj_type_hierarchy = self.get_object_types_hierarchy()

        # Get op param types
        single_obj_count = 0
        op_param_types = []
        op_param_supertypes = []
        for el in [el for el in op_params.strip().split() if el.strip() != "-"]:
            if el.startswith("?"):
                single_obj_count += 1
            else:
                [op_param_types.append([el]) if el not in obj_type_hierarchy.keys() else
                 op_param_types.append(obj_type_hierarchy[el])
                 for _ in range(single_obj_count)]

                [op_param_supertypes.append(el) for _ in range(single_obj_count)]
                single_obj_count = 0

        # op_param_types = [el for i,el in enumerate(op_params.split()) if (i+1)%3==0]

        # Get all predicates
        with open("PDDL/domain_learned.pddl", "r") as f:
            data = [el.strip() for el in f.read().split("\n")]
            preds = re.findall(":predicates.+?:action","".join(data))[0]

        all_predicates = sorted(re.findall("\([^()]*\)", preds))

        relevant_predicates = []

        for predicate in all_predicates:

            pred_name = predicate.strip()[1:-1].split()[0]
            # pred_param_types = [el for i,el in enumerate(predicate.strip()[1:-1].replace(pred_name, "").strip().split())
            #                if (i+1)%3 == 0]

            # Get predicate parameter types
            single_obj_count = 0
            pred_param_types = []
            pred_param_supertypes = []
            for el in [el for el in predicate.strip()[1:-1].strip().split()[1:] if el.strip() != "-"]:
                if el.startswith("?"):
                    single_obj_count += 1
                else:
                    [pred_param_types.append([el]) if el not in obj_type_hierarchy.keys()
                     else pred_param_types.append(obj_type_hierarchy[el])
                     for _ in range(single_obj_count)]

                    [pred_param_supertypes.append(el) for _ in range(single_obj_count)]

                    single_obj_count = 0

            # if all([pred_type in op_param_types for pred_type in pred_param_types]):
            # Check if predicate object types are contained into operator object types
            if all([any([el in [item for sublist in op_param_types for item in sublist]]
                        for el in pred_param_types[i]) for i in range(len(pred_param_types))]):



                all_pred_type_indices = []
                for pred_type in pred_param_types:
                    pred_type_indices = ["?param_{}".format(i+1)
                                         for i, op_pred_type in enumerate(op_param_types)
                                         if len([el for el in pred_type if el in op_pred_type]) > 0]
                                         # if op_pred_type == pred_type]
                    all_pred_type_indices.append(pred_type_indices)

                param_combinations = [list(p) for p in itertools.product(*all_pred_type_indices)]

                # Remove inconsistent combinations according to predicate input types
                param_comb_inconsistent = []
                for comb in param_combinations:

                    comb_param_types = []
                    for param in comb:
                        comb_param_types.append(op_param_supertypes[int(param.split("_")[1]) - 1])

                    for k, op_param_type in enumerate(comb_param_types):

                        if not ((pred_param_supertypes[k] in obj_type_hierarchy.keys() \
                            and op_param_type in obj_type_hierarchy[pred_param_supertypes[k]]) \
                            or op_param_type == pred_param_supertypes[k]):

                            param_comb_inconsistent.append(comb)

                            break

                # Remove inconsistent combinations
                [param_combinations.remove(comb) for comb in param_comb_inconsistent]


                if len(all_pred_type_indices) > 0:
                    relevant_predicates.extend(["({} {})".format(pred_name, " ".join(pred_comb))
                                                for pred_comb in param_combinations])
                else:
                    relevant_predicates.extend(["({})".format(pred_name)])

        return sorted(relevant_predicates)

    def get_op_relevant_predicatesONLYADMITDIFFERENTPARAMS(self, op_name):
    # def get_op_relevant_predicates(self, op_name):

        op_params = self.get_operator_parameters(op_name).strip()[1:-1]

        obj_type_hierarchy = self.get_object_types_hierarchy()

        # Get op param types
        single_obj_count = 0
        op_param_types = []
        op_param_supertypes = []
        for el in [el for el in op_params.strip().split() if el.strip() != "-"]:
            if el.startswith("?"):
                single_obj_count += 1
            else:
                [op_param_types.append([el]) if el not in obj_type_hierarchy.keys() else
                 op_param_types.append(obj_type_hierarchy[el])
                 for _ in range(single_obj_count)]

                [op_param_supertypes.append(el) for _ in range(single_obj_count)]
                single_obj_count = 0

        # op_param_types = [el for i,el in enumerate(op_params.split()) if (i+1)%3==0]

        # Get all predicates
        with open("PDDL/domain_learned.pddl", "r") as f:
            data = [el.strip() for el in f.read().split("\n")]
            preds = re.findall(":predicates.+?:action","".join(data))[0]

        all_predicates = sorted(re.findall("\([^()]*\)", preds))

        relevant_predicates = []

        for predicate in all_predicates:

            pred_name = predicate.strip()[1:-1].split()[0]
            # pred_param_types = [el for i,el in enumerate(predicate.strip()[1:-1].replace(pred_name, "").strip().split())
            #                if (i+1)%3 == 0]

            # Get predicate parameter types
            single_obj_count = 0
            pred_param_types = []
            pred_param_supertypes = []
            for el in [el for el in predicate.strip()[1:-1].strip().split()[1:] if el.strip() != "-"]:
                if el.startswith("?"):
                    single_obj_count += 1
                else:
                    [pred_param_types.append([el]) if el not in obj_type_hierarchy.keys()
                     else pred_param_types.append(obj_type_hierarchy[el])
                     for _ in range(single_obj_count)]

                    [pred_param_supertypes.append(el) for _ in range(single_obj_count)]

                    single_obj_count = 0

            # if all([pred_type in op_param_types for pred_type in pred_param_types]):
            # Check if predicate object types are contained into operator object types
            if all([any([el in [item for sublist in op_param_types for item in sublist]]
                        for el in pred_param_types[i]) for i in range(len(pred_param_types))]):



                all_pred_type_indices = []
                for pred_type in pred_param_types:
                    pred_type_indices = ["?param_{}".format(i+1)
                                         for i, op_pred_type in enumerate(op_param_types)
                                         if len([el for el in pred_type if el in op_pred_type]) > 0]
                                         # if op_pred_type == pred_type]
                    all_pred_type_indices.append(pred_type_indices)

                param_combinations = [list(p) for p in itertools.product(*all_pred_type_indices)]

                # Remove inconsistent combinations according to predicate input types
                param_comb_inconsistent = []
                for comb in param_combinations:

                    comb_param_types = []
                    for param in comb:
                        comb_param_types.append(op_param_supertypes[int(param.split("_")[1]) - 1])

                    for k, op_param_type in enumerate(comb_param_types):

                        if not ((pred_param_supertypes[k] in obj_type_hierarchy.keys() \
                            and op_param_type in obj_type_hierarchy[pred_param_supertypes[k]]) \
                            or op_param_type == pred_param_supertypes[k]):

                            param_comb_inconsistent.append(comb)

                            break

                # Remove inconsistent combinations
                [param_combinations.remove(comb) for comb in param_comb_inconsistent]


                if len(all_pred_type_indices) > 0:
                    # relevant_predicates.extend(["({} {})".format(pred_name, " ".join(pred_comb))
                    #                             for pred_comb in param_combinations])
                    relevant_predicates.extend(["({} {})".format(pred_name, " ".join(pred_comb))
                                                for pred_comb in param_combinations if len(pred_comb)==len(set(pred_comb))])
                else:
                    relevant_predicates.extend(["({})".format(pred_name)])

        return sorted(relevant_predicates)


    def get_false_relevant_predicates(self, action_label):

        operator = action_label.split("(")[0]

        a_params = action_label[action_label.find("(") + 1:action_label.find(")")].split(",")

        # Get current pddl state
        with open("PDDL/facts.pddl", "r") as f:
            data = [el.strip() for el in f.read().split("\n")]
            facts = re.findall(":init.*\(:goal","".join(data))[0]
        current_state_pddl = sorted(re.findall("\([^()]*\)", facts))

        with open("PDDL/domain_learned.pddl", 'r') as f:
            data = f.read().split("\n")

            objects_row = [el.replace(")", "").strip()
                           for el in
                           re.findall(":types.*\(:predicates", "++".join(data))[0].replace(":types", "").replace(
                               "(:predicates", "").split("++")
                           if el.strip() != ""]

            objects = defaultdict(list)

            start_index = 0
            for row in objects_row:
                current_index = len(objects['objects'])
                row = row.replace("(", "").replace(")", "")
                if row.find("- ") != -1:
                    [objects['objects'].append(el) for el in row.strip().split("- ")[0].split()]
                    objects[row.strip().split("- ")[1].strip()] = [el.strip()
                                                                  for el in
                                                                  row.strip().split("- ")[0].strip().split()]
                    # objects['objects'][start_index:current_index + 1]]
                    start_index = current_index + 1
                else:
                    [objects['objects'].append(el) for el in row.split()]

            for object_key, object_values in objects.items():
                if object_key != 'objects':

                    for val in object_values:

                        for key in objects.keys():
                            if val == key:
                                objects[object_key] = [el for el in objects[object_key] + objects[val] if el != val]

            action_separator_char = None
            check_syntax = False

            for j, line in enumerate(data):

                if data[j].strip().find('(:action') != -1 and not check_syntax:
                    if data[j].strip().split()[1].find("-") != -1:
                        action_separator_char = "-"
                        check_syntax = True

                    elif data[j].strip().split()[1].find("_") != -1:
                        action_separator_char = "_"
                        check_syntax = True

                if check_syntax:
                    operator_name_checked = operator.lower().replace('_', action_separator_char).replace('-',
                                                                                                         action_separator_char)
                else:
                    operator_name_checked = operator.lower()

                # if el.lower().startswith("(:action {}".format(operator_name_checked)):
                if line.lower().strip().find("action {}".format(operator_name_checked)) != -1:
                    # if el.lower().startswith("(:action {}".format(operator.replace("_","-").lower())):

                    tmp = data[j + 1][data[j + 1].find("parameters") + 10:-1] \
                        .replace("- ", "").replace(")", "").replace("(", "").split()

                    single_obj_count = 0
                    operator_objects = []
                    for el in tmp:
                        if el.startswith("?"):
                            single_obj_count += 1
                        else:
                            [operator_objects.append(el) for _ in range(single_obj_count)]
                            single_obj_count = 0




                    # operator_objects = [tmp[i] for i in range(len(tmp)) if not i % 2 == 0]
                    break

            all_predicates = re.findall("\([^()]*\)",re.findall(":predicates[^:]*\)\)",
                                                                  "".join([el.strip() for el in data]))[0])

        operator_objects_subtypes= [objects[el] if objects[el] != [] else el for el in operator_objects]
        all_param_objects_list = []
        all_param_objects_list.extend([" ".join(list(objects[obj])) if objects[obj] != [] else obj
                                       for obj in operator_objects])
        all_param_objects_list = list(set(" ".join(all_param_objects_list).split()))

        false_relevant_predicates = []
        for predicate in all_predicates:
            if len(predicate.split()) > 1:
                pred_name = predicate.split()[0][1:]
            else:
                pred_name = predicate[1:-1]

            pred_params = []
            # for line in predicate[:-1].split("?")[1:]:

            single_obj_count = 0
            for el in [el for el in predicate[:-1].split()[1:] if el.strip() != "-"]:
                if el.startswith("?"):
                    single_obj_count += 1
                else:

                    if len(objects[el]) == 0:
                        [pred_params.append([el]) for _ in range(single_obj_count)]
                    else:
                        [pred_params.append(objects[el]) for _ in range(single_obj_count)]

                    single_obj_count = 0



                # obj_type = el.split("-")[1].strip()
                # if len(objects[obj_type]) == 0:
                #     pred_params.append([obj_type])
                # else:
                #     pred_params.append(objects[obj_type])

            combinations = [list(p) for p in itertools.product(*pred_params)
                            # Check that all predicate objects are in operator input ones
                            if len([el for el in list(p) if el not in
                                    all_param_objects_list]) == 0]

            for elements in combinations:
                objects_count = []

                if len(elements) == 0:
                    false_relevant_predicates.append("({})".format(pred_name.strip()))

                else:

                    for line in elements:

                        seen_elements = []

                        for n in range(len(operator_objects_subtypes)):

                            if line == operator_objects_subtypes[n] or line in operator_objects_subtypes[n]:


                                if line not in seen_elements:
                                    seen_elements.append(line)
                                    objects_count.append([n])
                                else:
                                    # objects_count[seen_elements.index(el)].append(n)
                                    objects_count[-1].append(n)

                    objects_combinations = [list(p) for p in itertools.product(*objects_count)]

                    for combination in objects_combinations:

                        false_relevant_predicates.append(
                            "({} {})".format(pred_name, " ".join([a_params[el] for el in combination])))


        false_relevant_predicates = [pred for pred in false_relevant_predicates
                                     if pred not in [el.strip() for el in current_state_pddl]]

        false_relevant_predicates = sorted(list(set(false_relevant_predicates)))

        return false_relevant_predicates



    def get_dummy_domain_positive_effects(self):

        with open("PDDL/domain_dummy.pddl", 'r') as f:
            data = [el for el in f.read().split("\n") if el.strip() != ""]

            removed_rows = []

            for m in range(len(data)):

                row = data[m]

                if row.lower().strip().startswith("(:action"):

                    if data[m + 2].strip().startswith(":precondition") and data[m + 3].strip() == ")":
                        removed_rows.append(m)

                        for n in range(m + 1, len(data) - 1):
                            if data[n].strip().startswith("(:action"):
                                break
                            removed_rows.append(n)

            dummy_domain = [data[i] for i in range(len(data)) if i not in removed_rows]

        # Get dummy domain effects
        dummy_domain_effects = []
        for i in range(len(dummy_domain)):
            row = dummy_domain[i]

            if row.strip().lower().startswith(":effect"):

                start_index = i
                end_index = None

                for j in range(i, len(dummy_domain)):
                    if dummy_domain[j].strip().lower().startswith("(:action") or j == len(dummy_domain) - 1:
                        end_index = j
                        break

                # additive_effects = re.sub("\(not.*[^\)]\)\)", "",
                #                           " ".join(dummy_domain[start_index:end_index]))

                additive_effects = re.sub("\(not[^)]*\)\)", "",
                                          " ".join(dummy_domain[start_index:end_index]))

                dummy_domain_effects.extend([el.split()[0][1:].replace(")", "").replace("(", "")
                                             for el in re.findall("\([^()]*\)", additive_effects)])

                dummy_domain_effects = list(set(dummy_domain_effects))

        return dummy_domain_effects


    def get_dummy_domain_negative_effects(self):

        with open("PDDL/domain_dummy.pddl", 'r') as f:
            data = [el for el in f.read().split("\n") if el.strip() != ""]

            removed_rows = []

            for m in range(len(data)):

                row = data[m]

                if row.lower().strip().startswith("(:action"):

                    if data[m + 2].strip().startswith(":precondition") and data[m + 3].strip() == ")":
                        removed_rows.append(m)

                        for n in range(m + 1, len(data) - 1):
                            if data[n].strip().startswith("(:action"):
                                break
                            removed_rows.append(n)

            dummy_domain = [data[i] for i in range(len(data)) if i not in removed_rows]

        # Get dummy domain effects
        dummy_domain_negative_effects = []
        for i in range(len(dummy_domain)):
            row = dummy_domain[i]

            if row.strip().lower().startswith(":effect"):

                start_index = i
                end_index = None

                for j in range(i, len(dummy_domain)):
                    if dummy_domain[j].strip().lower().startswith("(:action") or j == len(dummy_domain) - 1:
                        end_index = j
                        break

                # additive_effects = re.sub("\(not[^)]*\)\)", "",
                #                           " ".join(dummy_domain[start_index:end_index]))

                negative_effects = re.findall("\(not[^)]*\)\)", " ".join(dummy_domain[start_index:end_index]))

                # dummy_domain_effects.extend([el.split()[0][1:].replace(")", "").replace("(", "")
                #                              for el in re.findall("\([^()]*\)", negative_effects)])

                dummy_domain_negative_effects.extend(negative_effects)

                dummy_domain_negative_effects = list(set(dummy_domain_negative_effects))

        return dummy_domain_negative_effects



    # Learn precondition of a not yet learned operator
    def learn_failed_action_precondition(self, simulator):

        not_learned_operators = [op for op in self.all_operators if op not in self.operator_learned]

        action_model_updated = False

        # Minimum number of precondition predicates to be tested in the planner subgoal
        n = 1

        false_relevant_predicates = defaultdict(list)

        operator_input_precs = defaultdict(list)

        for operator in not_learned_operators:
            false_relevant_predicates[operator] = self.get_op_relevant_predicates(operator)

            # If an input domain exists, remove certain preconditions
            operator_input_precs[operator] = sorted(self.get_operator_input_preconditions(operator))

            [false_relevant_predicates[operator].remove(prec) for prec in operator_input_precs[operator]]

        # Maximum number of precondition predicates to be tested
        max_pred = min(Configuration.MAX_PRECS_LENGTH,
                       max([len(v) for v in false_relevant_predicates.values()]))

        # if Configuration.INCREASING:
        #     # while not action_model_updated and n <= max_pred:
        #     condition = not action_model_updated and n <= max_pred
        # else:
        #     n = max_pred
        #     condition = not action_model_updated and n > 0
        #     # while not action_model_updated and n > 0:

        # while condition:
        while not action_model_updated and n <= max_pred:
            operator_possible_precs = dict()

            # Compute the binomial set (of length n) of possible combinations (with length n) of uncertain preconditions,
            # for each operator.
            for operator in not_learned_operators:
                operator_possible_precs[operator] = [sorted(list(el))
                                                      for el in itertools.chain.from_iterable(itertools.combinations(false_relevant_predicates[operator], r)
                                                                                              for r in range(n, n+1))
                                                     if sorted(list(el)) not in self.useless_possible_precs[operator]]


            for op_name in not_learned_operators:

                while not action_model_updated and len(operator_possible_precs[op_name])>0:

                    # Compute subgoal by adding the disjunction of all possible preconditions
                    # (e.g. subgoal = (p1 || p2 || p3 || p4 || p5))
                    op_params = self.get_operator_parameters(op_name)

                    subgoal = ""

                    for checked_precs in operator_possible_precs[op_name]:

                        # not_checked_precs = list(set([item for sublist in operator_possible_precs[op_name]
                        #                      for item in sublist if item not in checked_precs]))
                        #
                        # not_checked_precs = [el for el in not_checked_precs if el not in checked_precs]
                        #
                        # negated_precs = ["(not {})".format(el) for el in not_checked_precs]

                        subgoal += "(exists {} (and {}))\n".format(op_params,
                                                                   " ".join(operator_input_precs[op_name]
                                                                            # + negated_precs
                                                                            + checked_precs))

                    # DEBUG
                    print("\n\nChecking feasibility of operator {} with possible preconditions of length {}".format(op_name, n))

                    shutil.copyfile("PDDL/facts.pddl", os.path.join("PDDL/facts_dummy.pddl"))

                    with open("PDDL/facts_dummy.pddl", 'r') as f:
                        data = f.read().split("\n")
                        for i in range(len(data)):
                            row = data[i]

                            if row.strip().startswith("(:goal"):
                                end_index = i + 1
                                data[i] = "(:goal (or \n{}) \n))".format(subgoal)

                    with open("PDDL/facts_dummy.pddl", 'w') as f:
                        [f.write(el + "\n") for el in data[:end_index]]

                    plan, found = Planner.FD_dummy()
                    # plan, found = Planner.Madagascar("domain_dummy.pddl", "facts_dummy.pddl")

                    feasibility = plan is not None

                    # DEBUG
                    print("The feasibility of operator {} with possible preconditions of length {} is: {}".format(op_name, n, feasibility))

                    if not feasibility:
                        if n == 1:
                            [false_relevant_predicates[op_name].remove(prec[0]) for prec in operator_possible_precs[op_name]]
                        operator_possible_precs[op_name] = []

                    # Execute found plan
                    else:

                        for action in plan:

                            old_state = simulator.get_state()
                            obs, done = simulator.execute(action.lower())

                            # DEBUG
                            if done:
                                print("Successfully executed action {}: {}".format(self.action_labels.index(action.lower()),
                                                                                   action.lower()))

                                # Update action model effects
                                self.add_operator_effects(action.lower(), old_state, simulator.get_state())

                                self.tried_actions = []
                                self.not_executable_actions = []
                                self.not_executable_actions_index = []

                                # Evaluate online metrics for log file
                                new_now = default_timer()
                                self.time_at_iter.append(new_now - self.now)
                                self.now = new_now
                                self.iter += 1
                                self.eval_log()

                                self.last_action = self.action_labels.index(action.lower())

                                self.parser.update_pddl_facts(simulator.get_state())


                            else:

                                print('Something went wrong, the found plan should be executable, (otherwise it means the model'
                                      ' is not safe)')

                                exit()

                        # After plan has been executed, find actions of investigated operators which satisfy at least n
                        # preconditions

                        current_state = simulator.get_state()

                        considered_op_actions = []

                        sat_preconds = []

                        sat_preconds_placeholder = []

                        checked_op_precs = copy.deepcopy(operator_possible_precs[op_name])

                        # Warning: lento se la lista di azioni che iniziano con opname è molto numerosa (es. 20000)
                        for action in [action for k, action in enumerate(self.action_labels)
                                       if action.startswith(op_name + "(")]:
                                       # if action.startswith(op_name + "(") and k not in self.not_executable_actions_index]:

                            for precs in checked_op_precs:
                                precs = operator_input_precs[op_name] + precs
                                tmp_precs = copy.deepcopy(precs)
                                tmp_precs = "++".join(tmp_precs)
                                for j, act_obj in enumerate(action.split("(")[1].strip()[:-1].split(",")):
                                        tmp_precs = tmp_precs.replace("?param_{})".format(j+1), act_obj + ")")\
                                                     .replace("?param_{} ".format(j+1), act_obj + " ")
                                tmp_precs = tmp_precs.split("++")
                                action_sat_precs = [prec for prec in tmp_precs if prec in current_state]

                                if len(action_sat_precs) == n + len(operator_input_precs[op_name]):
                                    considered_op_actions.append(action)
                                    sat_preconds.append(action_sat_precs) # debug
                                    sat_preconds_placeholder.append(precs[len(operator_input_precs[op_name]):])

                            [checked_op_precs.remove(el) for el in sat_preconds_placeholder if el in checked_op_precs]


                        # Remove not executable actions according to executability constraints
                        self.not_executable_actions_index = self.compute_not_executable_actionsJAVA()

                        removed = []
                        # Warning: qui ci si impiega troppo tempo (es. len(consideredopactions)=5000)
                        for j, action in enumerate(considered_op_actions):
                            if self.action_labels.index(action) in self.not_executable_actions_index:
                                removed.append(j)

                                operator_possible_precs[op_name].remove(sat_preconds_placeholder[j])

                                if sorted(sat_preconds_placeholder[j]) not in self.useless_possible_precs[op_name]:
                                    self.useless_possible_precs[op_name].append(sorted(sat_preconds_placeholder[j]))

                                # # Check if preconditions of removed action appears again in the satisfied preconditions
                                # # list of the current state
                                # if sum([sat_preconds_placeholder[j] == el
                                #         for i,el in enumerate(sat_preconds_placeholder) if i not in removed]) == 0:
                                #
                                #     operator_possible_precs[op_name].remove(sat_preconds_placeholder[j])

                        removed.reverse()
                        for index in removed:
                            del(considered_op_actions[index])
                            del(sat_preconds[index])
                            del(sat_preconds_placeholder[index])




                        # Check considered action executions
                        removed = []
                        failed_actions = []
                        for j in range(len(considered_op_actions)):
                            # try to execute considered actions
                            action = considered_op_actions[j]

                            # Do not execute considered actions already failed
                            if action in failed_actions:
                            # if action in failed_actions or action not in self.action_labels:

                                if sat_preconds_placeholder[j] in operator_possible_precs[op_name]:
                                    operator_possible_precs[op_name].remove(sat_preconds_placeholder[j])

                                    if sorted(sat_preconds_placeholder[j]) not in self.useless_possible_precs[op_name]:
                                        self.useless_possible_precs[op_name].append(sorted(sat_preconds_placeholder[j]))

                                removed.append(j)
                            else:
                                sat_action_precs = sat_preconds_placeholder[j]

                                old_state = simulator.get_state()

                                obs, done = simulator.execute(action.lower())

                                # DEBUG
                                if done:
                                    # print("Successfully executed action {}: {}".format(a, self.action_labels[a]))
                                    print("Successfully executed action {}: {}".format(self.action_labels.index(action.lower()),
                                                                                       action.lower()))

                                    self.tried_actions = []
                                    self.not_executable_actions = []
                                    self.not_executable_actions_index = []

                                    # Update action model precondition
                                    self.changed_preconditions = self.add_operator_precondition(action)
                                    self.add_operator_effects(action, old_state, simulator.get_state())

                                    # Evaluate online metrics for log file
                                    new_now = default_timer()
                                    self.time_at_iter.append(new_now-self.now)
                                    self.now = new_now
                                    self.iter += 1
                                    self.eval_log()

                                    self.last_action = self.action_labels.index(action.lower())

                                    self.parser.update_pddl_facts(simulator.get_state())

                                    action_model_updated = True

                                    break


                                else:

                                    print("Not successfully executed action {}: {}".format(self.action_labels.index(action.lower()),
                                                                                       action.lower()))

                                    self.last_failed_action = action

                                    failed_actions.append(action)

                                    # Evaluate online metrics for log file
                                    new_now = default_timer()
                                    self.time_at_iter.append(new_now - self.now)
                                    self.now = new_now
                                    self.iter += 1
                                    self.eval_log()

                                    self.add_executability_constraint(action)

                                    if sat_action_precs in operator_possible_precs[op_name]:
                                        operator_possible_precs[op_name].remove(sat_action_precs)

                                    if sorted(sat_action_precs) not in self.useless_possible_precs[op_name]:
                                        self.useless_possible_precs[op_name].append(sorted(sat_action_precs))

                                    removed.append(j)

                        removed.reverse()
                        for index in removed:
                            del(considered_op_actions[index])
                            del(sat_preconds[index])
                            del(sat_preconds_placeholder[index])


            if not action_model_updated:

                # if Configuration.INCREASING:
                n = n + 1
                max_pred = min(Configuration.MAX_PRECS_LENGTH,
                               max([len(el) for el in false_relevant_predicates]))
                # else:
                #
                #     max_pred = max([len(el) for el in false_relevant_predicates])
                #
                #     n = n-1
                #     if max_pred < n:
                #         n = max_pred


        if not action_model_updated:
            # print('\n\n\nOperators {} cannot be learned for the first time, try to reach model convergence'
            #       ' with existing operators before investigating not learned ones.')
            # exit()
            print("\n\n\n\nModel convergence reached, although not all operators have been learned ")
            # Model convergence reached with, although not all operators have been learned
            return True



        return False




    def learn(self, eval_frequency=None, simulator=None):
        """
        Solve the problem instance
        :param max_iter: maximum number of iterations
        :param eval_frequency: iteration evaluation frequency
        :param simulator: domain simulator
        :return: None
        """

        np.random.seed(Configuration.RANDOM_SEED)

        self.max_time_limit = Configuration.TIME_LIMIT_SECONDS

        self.now = default_timer()
        self.initial_timer = default_timer()

        if eval_frequency is not None:
            self.eval_frequency = eval_frequency

        self.eval_log()

        # Get current state observation
        # obs, debug_info = simulator.reset()

        # Get all action list
        # self.all_environment_action_labels = [re.sub(":[^,]*(,|\))", ",", str(el))[:-1] + ")"
        #                                       for el in list(simulator.action_space._compute_all_ground_literals(obs))]
        # self.all_environment_actions = list(simulator.action_space._compute_all_ground_literals(obs))

        # Iterate for a maximum number of exploration steps
        for i in range(Configuration.MAX_ITER):

            # ####################### TEMP
            # if self.iter >= Configuration.MAX_ITER:
            #     print('Iteration limit reached ({} iterations)'.format(Configuration.MAX_ITER))
            #     if Configuration.RANDOM_WALK:
            #         self.add_not_learn_op_precs()
            #     break
            # ####################### TEMP

            # Check resolution time limit
            if default_timer()-self.initial_timer > self.max_time_limit:
                print('Time limit reached ({} seconds)'.format(self.max_time_limit))
                break

            # Check learned model convergence
            elif self.model_convergence:

                if not self.finalize_positive_effects_unknown(simulator) and not self.finalize_negative_effects_unknown(simulator) \
                        and not self.finalize_positive_effects_known(simulator) \
                        and not self.finalize_negative_effects_known(simulator):
                    print("\n\n\n\n\nModel convergence reached.")
                    break
                else:
                    self.model_convergence = False

            a = None

            # While the chosen action is illegal
            # while not self.model_convergence and default_timer()-self.initial_timer < self.max_time_limit:
            ####################### TEMP
            while not self.model_convergence and default_timer()-self.initial_timer < self.max_time_limit \
                    and self.iter < Configuration.MAX_ITER:
            ####################### TEMP

                # Choose an action
                a, applied_strategy = self.select_action()

                # If no precondition refinements can be done and all operator has already been learned, try to change
                # state in order to test a single precondition (uncertain) predicate of a random operator
                while a is None:

                    self.model_convergence = self.guide_agent_to_exploration_state(simulator)

                    if self.model_convergence:
                        break

                    # Choose an action
                    a, applied_strategy = self.select_action()


                # Check model convergence
                if self.model_convergence:
                    break


                # Execute the action through the domain simulator
                # action_index = self.all_environment_action_labels.index(self.action_labels[a])
                # action = self.all_environment_actions[action_index]
                # old_state = simulator.get_state()
                # obs, reward, done, debug_info = simulator.step(action)

                old_state = simulator.get_state()
                obs, done = simulator.execute(self.action_labels[a])

                # if obs != old_state or self.action_labels[a] in self.executable_actions:
                #     done = True
                # else:
                #     done = False

                # The action has been successfully executed
                if done:
                    print("Successfully executed action {}: {}".format(a, self.action_labels[a]))

                    self.tried_actions = []
                    self.not_executable_actions = []
                    self.not_executable_actions_index = []

                    if applied_strategy != Configuration.STRATEGIES[0]:
                        self.current_plan = None

                    # Update action model precondition
                    self.changed_preconditions = self.add_operator_precondition(self.action_labels[a])
                    self.add_operator_effects(self.action_labels[a], old_state, simulator.get_state())

                    # Evaluate online metrics for log file
                    new_now = default_timer()
                    self.time_at_iter.append(new_now-self.now)
                    self.now = new_now
                    self.iter += 1
                    self.eval_log()

                    # Update pddl state
                    self.parser.update_pddl_facts(simulator.get_state())

                    self.executable_actions = self.compute_executable_actions()

                # The action has not been successfully executed
                else:
                    print("Not Successfully executed action {}: {}".format(a, self.action_labels[a]))

                    self.last_failed_action = self.action_labels[a]

                    # Evaluate online metrics for log file
                    new_now = default_timer()
                    self.time_at_iter.append(new_now - self.now)
                    self.now = new_now
                    self.iter += 1
                    self.eval_log()

                    # Store action objects predicates conjunction
                    action_label = self.action_labels[a]

                    self.add_executability_constraint(action_label)

                    # self.add_positive_precondition(action_label)

                    # Check for new certain preconditions, i.e., when an action cannot be executed and there is only
                    # one false precondition, then the false precondition is stored as a certain one
                    operator = action_label.split("(")[0]
                    a_params = action_label.split("(")[1][:-1].split(",")

                    with open("PDDL/facts.pddl", "r") as f:
                        data = [el.strip() for el in f.read().split("\n")]
                        facts = re.findall(":init.*\(:goal", "".join(data))[0]
                    current_state_pddl = sorted(re.findall("\([^()]*\)", facts))

                    action_preconditions = []
                    for checked_pred in self.operator_certain_predicates[operator]:

                        for k in range(len(a_params)):
                            checked_pred = checked_pred.replace("?param_{}".format(k+1), a_params[k])

                        action_preconditions.append(checked_pred)

                    if len([precond for precond in action_preconditions if precond in current_state_pddl]) \
                        == len(self.operator_certain_predicates[operator]):

                        action_preconditions = []

                        for checked_pred in self.operator_uncertain_predicates[operator]:

                            for k in range(len(a_params)):
                                checked_pred = checked_pred.replace("?param_{} ".format(k + 1), a_params[k] + " ")
                                checked_pred = checked_pred.replace("?param_{})".format(k + 1), a_params[k] + ")")

                            action_preconditions.append(checked_pred)

                        not_satisfied_preconds = [precond for precond in action_preconditions
                                                 if precond not in current_state_pddl]

                        if len(not_satisfied_preconds) == 1:

                            # not_satisfied_precond = not_satisfied_preconds[0]
                            #
                            # for k in range(len(a_params)):
                            #     not_satisfied_precond = not_satisfied_precond.replace(a_params[k] + " ",
                            #                                                           "?param_{} ".format(k + 1))
                            #     not_satisfied_precond = not_satisfied_precond.replace(a_params[k] + ")",
                            #                                                           "?param_{})".format(k + 1))

                            not_satisfied_precond = self.operator_uncertain_predicates[operator]\
                            [action_preconditions.index(not_satisfied_preconds[0])]

                            self.operator_uncertain_predicates[operator].remove(not_satisfied_precond)
                            self.operator_certain_predicates[operator].append(not_satisfied_precond)

                            # DEBUG
                            print("operator {}, adding certain precondition: {}".format(operator,
                                                                                        not_satisfied_precond))

                            # Add executability constraint, if a certain precondition is false then the action
                            # cannot be selected
                            for constr in self.operator_executability_constr[operator]:
                                if set(["not(" + not_satisfied_precond + ")"]).issubset(set(constr)):
                                    self.operator_executability_constr[operator].remove(constr)

                            self.operator_executability_constr[operator].append(["not(" + not_satisfied_precond + ")"])




                    # Update not executable actions, this may be necessary after having reduced the overall action
                    # labels list
                    # self.not_executable_actions_index = self.compute_not_executable_actionsJAVA()


                    if len(self.executable_actions) + len(self.not_executable_actions_index) == len(self.action_labels):

                        # Maximally refine learned operators (before trying to learn a new one)
                        print("\n\nRefining learned operators")
                        convergence = False
                        while not convergence:
                            convergence = self.guide_agent_to_exploration_state(simulator)

                        print("\n\n Trying to learn a new operator")
                        # Try to learn a new operator
                        self.model_convergence = self.learn_failed_action_precondition(simulator)

                        # executable = self.learn_failed_action_precondition(self.action_labels[a], simulator)
                        #
                        # if not executable:
                        #     # If an FF action cannot be executed, then invalid current FF plan
                        #     if applied_strategy == Configuration.STRATEGIES[1]:
                        #         self.current_plan = None
                        # else:
                        #     print("Successfully executed action {}: {}".format(a, self.action_labels[a]))
                        #
                        #     # Update action model precondition
                        #     self.changed_preconditions = self.add_operator_precondition(self.action_labels[a])
                        #
                        #     # Evaluate online metrics for log file
                        #     new_now = default_timer()
                        #     self.time_at_iter.append(new_now-self.now)
                        #     self.now = new_now
                        #     self.iter += 1
                        #     self.eval_log()
                        #
                        #     self.parser.update_pddl_facts(simulator.get_state())
                        #
                        #     # Update executable action list
                        #     if self.current_plan is None:
                        #         self.executable_actions = self.compute_executable_actions()
                        #         executable_actions = [self.action_labels.index(el) for el in
                        #                               self.executable_actions]
                        #         self.action_precond_perc_filtered = [self.action_precondition_percentage[i]
                        #                                              if i not in executable_actions + self.tried_actions else -1
                        #                                              for i in
                        #                                              range(len(self.action_precondition_percentage))]
                        #         self.not_executable_actions_index = []


                self.checking_precondition = False
                self.checked_precondition = None

                # if applied_strategy == Configuration.STRATEGIES[1] and self.current_plan is None:
                if self.goal_reached():
                    goal_achieved = True

            # Check resolution time limit
            if default_timer()-self.initial_timer > self.max_time_limit:
                print('Time limit reached ({} seconds)'.format(self.max_time_limit))
                break


            # ####################### TEMP
            # if self.iter >= Configuration.MAX_ITER:
            #     print('Iteration limit reached ({} iterations)'.format(Configuration.MAX_ITER))
            #     if Configuration.RANDOM_WALK:
            #         self.add_not_learn_op_precs()
            #     break
            # ####################### TEMP


            self.last_action = a

            # Update pddl state
            self.parser.update_pddl_facts(simulator.get_state())

            # Update executable action list
            # if self.current_plan is None:
            self.executable_actions = self.compute_executable_actions()
            executable_actions = [self.action_labels.index(el) for el in self.executable_actions]
            self.action_precond_perc_filtered = [self.action_precondition_percentage[i]
                                            if i not in executable_actions + self.tried_actions else -1
                                            for i in range(len(self.action_precondition_percentage))]

            self.not_executable_actions_index = []

        # Evaluate online metrics for log file
        new_now = default_timer()
        self.time_at_iter.append(new_now - self.now)
        self.now = new_now
        self.iter += 1
        self.eval_log()


        # Remove uncertain negative effects which are not in the preconditions (strips assumption)
        if Configuration.NEG_EFF_ASSUMPTION:
            for op in self.operator_learned:
                op_precond = self.get_operator_preconditions(op)
                self.uncertain_negative_effects[op] = [pred for pred in self.uncertain_negative_effects[op]
                                                       if pred in op_precond]

        print("-------------- METRICS WITH UNCERTAIN NEGATIVE EFFECTS --------------")
        self.eval_log_with_uncertain_neg()


    def create_facts_and_domain_test(self, op, neg_precs):

        shutil.copyfile("PDDL/domain_dummy.pddl", "PDDL/domain_test.pddl")
        shutil.copyfile("PDDL/facts.pddl", "PDDL/facts_test.pddl")

        # Create facts file with fictitious goal
        with open("PDDL/facts_test.pddl", "r") as f:
            data = f.read().split("\n")

        with open("PDDL/facts_test.pddl", "w") as f:

            for i in range(len(data)):

                if data[i].find(":goal") != -1:

                    for j in range(i, len(data)):
                        data[j] = ""

                    data[i] = "(:goal (and (true))))"

            [f.write(el + "\n") for el in data]


        # Create domain file with tested operators, each tested operator has some negated preconditions
        with open("PDDL/domain_test.pddl", "r") as f:
            data = f.read().split("\n")

        with open("PDDL/domain_test.pddl", "w") as f:

            for i in range(len(data)):

                if data[i].find("(:action") != -1:
                    if data[i].strip().split()[1] == op:
                        data[i] = ""
                        data[i+1] = ""
                        data[i+2] = ""
                        data[i+3] = ""
                        data[i+4] = ""
                        data[i+5] = ""
                        break

            for i in range(len(data)):

                if data[i].find("(:predicates") != -1:
                    data[i] = data[i] + "\n(true )"

                if data[i].find("(:action") != -1:

                    test_operators = ""

                    op_precs = self.operator_certain_predicates[op] + self.operator_uncertain_predicates[op]
                    op_params = self.get_operator_parameters(op)
                    op_effect = self.get_operator_effects(op)

                    for k, neg_prec in enumerate(neg_precs):
                        test_op_precs = [p for p in op_precs if p not in neg_prec]
                        [test_op_precs.append("(not {})".format(p)) for p in neg_prec]

                        test_operators += "\n(:action {}-test-{}\n:parameters {}\n:precondition (and {})\n:effect (and (true ) {}))"\
                            .format(op, k, op_params, " ".join(test_op_precs), op_effect)

                    data[i] = test_operators + "\n" + data[i]
                    break

            [f.write(row + "\n") for row in data]


    def get_operator_effects(self, op):

        with open("PDDL/domain_learned.pddl", "r") as f:
            data = f.read().split("\n")

            op_effect = ""

            for i in range(len(data)):
                if data[i].strip() == "(:action {}".format(op):
                    for j in range(i+1, len(data)):
                        if data[j].find(":effect") != -1:
                            for k in range(j+1,len(data)):
                                if data[k].find(":actions"):
                                    op_effect = "\n".join(data[j:k])
                                    break

                            if op_effect == "":
                                op_effect = "\n".join(data[j:len(data)])
                            break
                    break

        op_effect = op_effect.replace(":effect", "").replace("(and", "").strip()[:-2]

        return op_effect


    def guide_agent_to_exploration_stateFICTOPERATORS(self, simulator):

        # Minimum number of precondition predicates to be tested in the planner subgoal
        n = 1

        # Maximum number of precondition predicates to be tested
        max_pred = max([len(preds) for op, preds in self.operator_uncertain_predicates.items()])

        all_operator_uncertain_preds = copy.deepcopy(dict(self.operator_uncertain_predicates))

        operator_uncertain_preds = dict()

        # Compute the binomial set (of length n) of possible combinations (with length n) of uncertain preconditions,
        # for each operator.
        for operator in all_operator_uncertain_preds:
            operator_uncertain_preds[operator] = [sorted(list(el))
                                                  for el in itertools.chain.from_iterable(itertools.combinations(all_operator_uncertain_preds[operator], r)
                                                                                          for r in range(n, n+1))]


        # Operator preconditions which cannot be checked because of planner failure (i.e. the planner cannot find a
        # plan)
        operator_unfeasible_precs = defaultdict(list)

        reset_operators = False

        while n <= max_pred:

            for operator in all_operator_uncertain_preds:

                if reset_operators:
                    reset_operators = False
                    n = 0
                    break

                # If there are no preconditions of length n to check
                if len(operator_uncertain_preds[operator]) > 0:

                    print("Checking uncertain preconditions of operator {} with length {}".format(operator, n))

                    self.create_facts_and_domain_test(operator, operator_uncertain_preds[operator])

                    # plan_exists = True

                    plan_failed = None

                    while True:

                        plan, found = Planner.FD_test()
                        # plan, found = Planner.Madagascar("domain_test.pddl", "facts_test.pddl")

                        if not found:
                            break

                        # Execute plan
                        plan_failed = False
                        for j, action in enumerate(plan):

                            if action.lower().find("-test-") == -1:
                                action = action.lower()
                            else:
                                action = action[:action.lower().find("-test-")].lower() + action[action.find("("):].lower()

                            a = self.action_labels.index(action)

                            old_state = simulator.get_state()
                            obs, done = simulator.execute(self.action_labels[a])

                            # The action has been successfully executed
                            if done:
                                print("Successfully executed action {}: {}".format(a, self.action_labels[a]))

                                self.tried_actions = []
                                self.not_executable_actions = []
                                self.not_executable_actions_index = []

                                # Update action model precondition
                                self.changed_preconditions = self.add_operator_precondition(self.action_labels[a])

                                # Update action model effects
                                self.add_operator_effects(self.action_labels[a], old_state, simulator.get_state())

                                # Evaluate online metrics for log file
                                new_now = default_timer()
                                self.time_at_iter.append(new_now-self.now)
                                self.now = new_now
                                self.iter += 1
                                self.eval_log()

                                # Update pddl state
                                self.parser.update_pddl_facts(simulator.get_state())

                                # self.executable_actions = self.compute_executable_actions()

                            # The action has not been successfully executed
                            else:
                                print("Not Successfully executed action {}: {}".format(a, self.action_labels[a]))

                                plan_failed = True

                                self.last_failed_action = self.action_labels[a]

                                # Evaluate online metrics for log file
                                new_now = default_timer()
                                self.time_at_iter.append(new_now - self.now)
                                self.now = new_now
                                self.iter += 1
                                self.eval_log()

                                # Store action objects predicates conjunction
                                action_label = self.action_labels[a]

                                self.add_executability_constraint(action_label)

                                # Check for new certain preconditions, i.e., when an action cannot be executed and there is only
                                # one false precondition, then the false precondition is stored as a certain one
                                operator = action_label.split("(")[0]
                                a_params = action_label.split("(")[1][:-1].split(",")

                                with open("PDDL/facts.pddl", "r") as f:
                                    data = [el.strip() for el in f.read().split("\n")]
                                    facts = re.findall(":init.*\(:goal", "".join(data))[0]
                                current_state_pddl = sorted(re.findall("\([^()]*\)", facts))

                                action_preconditions = []
                                for checked_pred in self.operator_certain_predicates[operator]:

                                    for k in range(len(a_params)):
                                        checked_pred = checked_pred.replace("?param_{}".format(k+1), a_params[k])

                                    action_preconditions.append(checked_pred)

                                if len([precond for precond in action_preconditions if precond in current_state_pddl]) \
                                    == len(self.operator_certain_predicates[operator]):

                                    action_preconditions = []

                                    for checked_pred in self.operator_uncertain_predicates[operator]:

                                        for k in range(len(a_params)):
                                            checked_pred = checked_pred.replace("?param_{} ".format(k + 1), a_params[k] + " ")
                                            checked_pred = checked_pred.replace("?param_{})".format(k + 1), a_params[k] + ")")

                                        action_preconditions.append(checked_pred)

                                    not_satisfied_preconds = [precond for precond in action_preconditions
                                                             if precond not in current_state_pddl]

                                    if len(not_satisfied_preconds) == 1:

                                        not_satisfied_precond = self.operator_uncertain_predicates[operator]\
                                        [action_preconditions.index(not_satisfied_preconds[0])]

                                        self.operator_uncertain_predicates[operator].remove(not_satisfied_precond)
                                        self.operator_certain_predicates[operator].append(not_satisfied_precond)

                                        # DEBUG
                                        print("operator {}, adding certain precondition: {}".format(operator,
                                                                                                    not_satisfied_precond))

                                        # Add executability constraint, if a certain precondition is false then the action
                                        # cannot be selected
                                        for constr in self.operator_executability_constr[operator]:
                                            if set(["not(" + not_satisfied_precond + ")"]).issubset(set(constr)):
                                                self.operator_executability_constr[operator].remove(constr)

                                        self.operator_executability_constr[operator].append(["not(" + not_satisfied_precond + ")"])


                                # operator_uncertain_preds[operator].remove()
                                failed_op = plan[j].strip().lower()[:plan[j].strip().find("(")]
                                op_neg_precs = [p.replace("(not ", "").strip()[:-1]
                                                for p in self.get_operator_test_preconditions_neg(failed_op)]
                                operator_uncertain_preds[operator].remove(op_neg_precs)
                                self.create_facts_and_domain_test(operator, operator_uncertain_preds[operator])

                        # If plan has not failed, then action model has been updated
                        if not plan_failed:
                            break

                    if plan_failed is not None and not plan_failed:
                        reset_operators = True

            n = n + 1

            for operator in all_operator_uncertain_preds:
                operator_uncertain_preds[operator] = [sorted(list(el))
                                                      for el in itertools.chain.from_iterable(
                        itertools.combinations(all_operator_uncertain_preds[operator], r)
                        for r in range(n, n + 1))]


        # Model learning has converged with learned operators
        return True



    def guide_agent_to_exploration_state(self, simulator):

        # Minimum number of precondition predicates to be tested in the planner subgoal
        n = 1

        # Maximum number of precondition predicates to be tested
        max_pred = max([len(preds) for op, preds in self.operator_uncertain_predicates.items()])

        all_operator_uncertain_preds = copy.deepcopy(dict(self.operator_uncertain_predicates))

        operator_uncertain_preds = dict()

        # Compute the binomial set (of length n) of possible combinations (with length n) of uncertain preconditions,
        # for each operator.
        for operator in all_operator_uncertain_preds:
            operator_uncertain_preds[operator] = [sorted(list(el))
                                                  for el in itertools.chain.from_iterable(itertools.combinations(all_operator_uncertain_preds[operator], r)
                                                                                          for r in range(n, n+1))
                                                  if sorted(list(el)) not in self.useless_negated_precs[operator]]


        # Operator preconditions which cannot be checked because of planner failure (i.e. the planner cannot find a
        # plan)
        operator_tested_precs = defaultdict(list)

        plan = None

        # Look for a plan to guide the agent in an exploration state (i.e. a state where the agent can try and
        # test a single precondition of an operator)

        while plan is None:

            for op in self.operator_learned:
                # Remove preconditions which cannot be checked because no plan can be computed or the corresponding
                # action is not executable (according to either executability constraints or previous execution failure)
                for prec in operator_tested_precs[op]:
                    if sorted(prec) in operator_uncertain_preds[op]:
                        operator_uncertain_preds[op].remove(sorted(prec))

            # Remove precondition combinations which contain (new) certain predicates
            for op in self.operator_learned:
                to_remove = []
                for prec in operator_uncertain_preds[op]:
                    if len([el for el in prec if el in self.operator_certain_predicates[op]]) != 0:
                        to_remove.append(prec)

                operator_uncertain_preds[op] = [prec for prec in operator_uncertain_preds[op] if prec not in to_remove]

            refined_operator = None

            while refined_operator is None and n <= max_pred:

                # Check feasibility
                for op in sorted(self.operator_learned):
                    negated_op_uncertain_precs = [el for el in operator_uncertain_preds[op] if len(el) == n]

                    if len(negated_op_uncertain_precs) != 0:
                        plan, feasible = self.check_feasibile_preconditions_negation(op, negated_op_uncertain_precs)

                        if not feasible:
                            [operator_uncertain_preds[op].remove(prec) for prec in negated_op_uncertain_precs]
                        else:
                            refined_operator = op
                            break

                # # Select an operator to be refined
                if refined_operator is None:

                    n = n + 1

                    for operator in self.operator_learned:
                        operator_uncertain_preds[operator] = [sorted(list(el))
                                                              for el in itertools.chain.from_iterable(itertools.combinations(all_operator_uncertain_preds[operator], r)
                                                                                                      for r in range(n, n+1))
                                                              if sorted(list(el)) not in self.useless_negated_precs[operator]]

            # Model convergence reached
            if refined_operator is None:
                # print('\n\n\n\n\n\nModel convergence reached')
                return True

            effects_updated = False
            for action in plan:

                # a = self.action_labels.index(action.lower())
                # action_index = self.all_environment_action_labels.index(self.action_labels[a])
                # action = self.all_environment_actions[action_index]
                #
                # old_state = simulator.get_state()
                # obs, reward, done, debug_info = simulator.step(action)

                old_state = simulator.get_state()
                obs, done = simulator.execute(action.lower())

                # if obs != old_state:
                #     done = True
                # else:
                #     done = False

                # DEBUG
                if done:
                    # print("Successfully executed action {}: {}".format(a, self.action_labels[a]))
                    print("Successfully executed action {}: {}".format(self.action_labels.index(action.lower()),
                                                                       action.lower()))

                    # Update action model effects
                    effects_updated = self.add_operator_effects(action.lower(), old_state, simulator.get_state())

                    self.tried_actions = []
                    self.not_executable_actions = []
                    self.not_executable_actions_index = []

                    # Evaluate online metrics for log file
                    new_now = default_timer()
                    self.time_at_iter.append(new_now - self.now)
                    self.now = new_now
                    self.iter += 1
                    self.eval_log()

                    self.last_action = self.action_labels.index(action.lower())

                    self.parser.update_pddl_facts(simulator.get_state())

                    # Some new negative effects may conflicts with next plan actions preconditions
                    if effects_updated:
                        break


                else:

                    print('Something went wrong, the found plan should be executable, (otherwise it means the model'
                          ' is not safe)')

                    exit()

            if effects_updated:
                plan = None

            #plan is no more none
            if plan is not None:
                # Compute not executable actions
                self.not_executable_actions_index = self.compute_not_executable_actionsJAVA()

                # Compute executable actions
                self.executable_actions = self.compute_executable_actions()

                self.action_precond_perc_filtered = [el for el in self.action_precondition_percentage]

                for el in self.not_executable_actions_index:
                    self.action_precond_perc_filtered[el] = -1

                for el in [self.action_labels.index(a) for a in self.executable_actions]:
                    self.action_precond_perc_filtered[el] = -1

                considered_op_actions = []
                unsat_precs_of_op_actions = []

                # Compute satisfied preconditions number of refined operator actions
                current_state = simulator.get_state()

                op_precs = sorted(self.get_operator_preconditions(refined_operator))

                for action_label in [action for action in self.action_labels if action.startswith(refined_operator + "(")]:
                    action_precs = self.get_action_positive_preconditions(action_label)

                    tmp_not_sat_precs = [prec for prec in action_precs if prec not in current_state]
                    not_sat_precs = []
                    for prec in tmp_not_sat_precs:
                        not_sat_precs.append(op_precs[action_precs.index(prec)])
                        action_precs[action_precs.index(prec)] = None
                    # not_sat_precs = [op_precs[action_precs.index(prec)] for prec in not_sat_precs]
                    not_sat_precs_len = len(not_sat_precs)

                    if not_sat_precs_len == n and \
                            len([prec for prec in not_sat_precs if prec in self.operator_certain_predicates[refined_operator]]) == 0:
                        considered_op_actions.append(action_label)
                        unsat_precs_of_op_actions.append(not_sat_precs)

                # Check if there is an action which does not satisfy exactly n preconditions but is already known to be not
                # executable due to some executability constraint (in this case, if the not satisfied precondition is a single
                # one, then it is a certain precondition).
                # The potentially executable action which is known to be not executable due to an executability constraint
                # must be removed by the potentially executable actions list.
                removed_actions_indices = []
                for i in range(len(considered_op_actions)):
                    action_label = considered_op_actions[i]

                    if self.action_labels.index(action_label) in self.not_executable_actions_index:

                        self.operator_executability_constr[refined_operator].append(["not({})".format(prec)
                                                                                     for prec in unsat_precs_of_op_actions[i]])
                        self.update_executability_constr(refined_operator, op_precs)

                        operator_tested_precs[refined_operator].append(unsat_precs_of_op_actions[i])

                        if unsat_precs_of_op_actions[i] not in self.useless_negated_precs[refined_operator]:
                            self.useless_negated_precs[refined_operator].append(sorted(unsat_precs_of_op_actions[i]))

                        removed_actions_indices.append(i)


                removed_actions_indices.reverse()
                for i in removed_actions_indices:
                    del considered_op_actions[i]
                    del unsat_precs_of_op_actions[i]

                # Select a potentially executable action and try to execute it

                if len(considered_op_actions) > 0:

                    action = considered_op_actions[0]
                    unsat_action_precs = unsat_precs_of_op_actions[0]

                    obs, done = simulator.execute(action.lower())

                    # DEBUG
                    if done:
                        # print("Successfully executed action {}: {}".format(a, self.action_labels[a]))
                        print("Successfully executed action {}: {}".format(self.action_labels.index(action.lower()),
                                                                           action.lower()))

                        self.tried_actions = []
                        self.not_executable_actions = []
                        self.not_executable_actions_index = []

                        # Update action model precondition
                        self.changed_preconditions = self.add_operator_precondition(action)

                        # Evaluate online metrics for log file
                        new_now = default_timer()
                        self.time_at_iter.append(new_now-self.now)
                        self.now = new_now
                        self.iter += 1
                        self.eval_log()

                        self.last_action = self.action_labels.index(action.lower())

                        self.parser.update_pddl_facts(simulator.get_state())


                    else:

                        print("Not successfully executed action {}: {}".format(self.action_labels.index(action.lower()),
                                                                           action.lower()))

                        self.last_failed_action = action

                        # Evaluate online metrics for log file
                        new_now = default_timer()
                        self.time_at_iter.append(new_now - self.now)
                        self.now = new_now
                        self.iter += 1
                        self.eval_log()

                        self.operator_executability_constr[refined_operator].append(["not({})".format(prec)
                                                                                     for prec in unsat_action_precs])
                        self.update_executability_constr(refined_operator, op_precs)

                        operator_tested_precs[refined_operator].append(unsat_action_precs)

                        if unsat_action_precs not in self.useless_negated_precs[refined_operator]:
                            self.useless_negated_precs[refined_operator].append(sorted(unsat_action_precs))

                        plan = None

                else:
                    plan = None

        # Model learning has not yet converged
        return False




    def add_executability_constraint(self, action_label):

        operator = action_label.strip().split("(")[0]

        op_precond = []
        if operator in self.operator_learned:
            op_precond = self.get_operator_preconditions(operator)

        negative_preconditions = self.get_negative_precondition(action_label)
        # positive_preconditions = self.get_op_positive_precondition(action_label)

        if len(op_precond) > 0:
            negative_preconditions = [el for el in negative_preconditions if el in op_precond]
            # positive_preconditions = [el for el in positive_preconditions if el in op_precond]

        negative_preconditions = [ "not({})".format(el) for el in negative_preconditions]

        # Check redundant constraints
        for constr in self.operator_executability_constr[operator]:
            # if set(negative_preconditions + positive_preconditions).issubset(set(constr)):
            if set(negative_preconditions).issubset(set(constr)):
                self.operator_executability_constr[operator].remove(constr)

        # self.operator_executability_constr[operator].append(negative_preconditions + positive_preconditions)

        self.operator_executability_constr[operator].append(negative_preconditions)


        # Filter action label list by removing all illegal actions according to the last computed executability constraint
        # self.compute_illegal_actions(operator, negative_preconditions)


    def get_operator_test_preconditions_neg(self, operator):

        with open("PDDL/domain_test.pddl", "r") as f:
            data = [el.strip() for el in f.read().split("\n")]

            all_action_schema = " ".join(data)[" ".join(data).index(":action"):]
            # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
            action_schema = re.findall(":action {}(.*?):effect".format(operator), all_action_schema)[0]
            preconds = sorted(re.findall("\(not \([^()]*\)\)", action_schema[action_schema.find("precondition"):]))

        return preconds


    def get_operator_preconditions(self, operator):

        with open("PDDL/domain_dummy.pddl", "r") as f:
            data = [el.strip() for el in f.read().split("\n")]

            all_action_schema = " ".join(data)[" ".join(data).index(":action"):]
            # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
            action_schema = re.findall(":action {}(.*?):effect".format(operator), all_action_schema)[0]
            preconds = sorted(re.findall("\([^()]*\)", action_schema[action_schema.find("precondition"):]))

        return preconds


    def get_operator_input_preconditions(self, operator):

        if not os.path.exists("PDDL/domain_input.pddl"):
            return []

        with open("PDDL/domain_input.pddl", "r") as f:
            data = [el.strip() for el in f.read().split("\n")]

            all_action_schema = " ".join(data)[" ".join(data).index(":action"):]
            # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
            action_schema = re.findall(":action {}(.*?):effect".format(operator), all_action_schema)[0]
            preconds = sorted(re.findall("\([^()]*\)", action_schema[action_schema.find("precondition"):]))

        preconds = [p for p in preconds if p.strip() != "(and )"]

        return preconds


    def get_operator_parameters(self, operator):

        with open("PDDL/domain_learned.pddl", "r") as f:
            data = [el.strip() for el in f.read().split("\n")]

            all_action_schema = " ".join(data)[" ".join(data).index(":action"):]
            # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
            action_schema = re.findall(":action {}(.*?):effect".format(operator), all_action_schema)[0]
            op_params = re.findall(":parameters(.*?):precondition", action_schema)[0].strip()

        return op_params




    def get_negative_precondition(self, action_label):

        operator = action_label.strip().split("(")[0]
        a_params = action_label[action_label.find("(") + 1:action_label.find(")")].split(",")
        # negative_preconditions = self.get_true_relevant_predicates(action_label)
        negative_preconditions = self.get_false_relevant_predicates(action_label)
        negative_preconditions_placeholder = []
        params_placeholder = defaultdict(list)
        checked_params = []
        for el in a_params:

            if el not in checked_params:
                el_indexes = [i for i in range(len(a_params)) if a_params[i] == el]

                for index in el_indexes:
                    params_placeholder[el].append("?param_{}".format(index+1))

                checked_params.append(el)

        for pred in negative_preconditions:
            pred_params = pred[1:-1].split()[1:]

            pred_params_placeholder = [params_placeholder[obj] for obj in pred_params]

            var_params_combinations = [list(p) for p in itertools.product(*pred_params_placeholder)]

            for tup in var_params_combinations:

                newpred = pred

                for k, subst in enumerate(tup):

                    if newpred.find(pred_params[k] + " ") != -1:
                        newpred = newpred.replace(pred_params[k] + " ", subst + " ", 1)
                    elif newpred.find(pred_params[k] + ")") != -1:
                        newpred = newpred.replace(pred_params[k] + ")", subst + ")", 1)
                    else:
                        print("Something went wrong when replacing objects with parameters placeholder "
                              "in get_negative_precondition()")

                negative_preconditions_placeholder.append(newpred)

        # # Check redundant negative preconditions
        # for precond in self.operator_negative_preconditions[operator]:
        #     if set(negative_preconditions_placeholder).issubset(set(precond)):
        #         self.operator_negative_preconditions[operator].remove(precond)

        return negative_preconditions_placeholder


    def get_true_relevant_predicates(self, action_label):

        a_params = action_label[action_label.find("(") + 1:action_label.find(")")].split(",")

        with open("PDDL/facts.pddl", "r") as f:
            data = [el.strip() for el in f.read().split("\n")]
            facts = re.findall(":init.*\(:goal","".join(data))[0]

        current_state_pddl = sorted(re.findall("\([^\(\)]*\)", facts))

        true_relevant_facts = []

        for pred in current_state_pddl:

            pred_params = pred[1:-1].split()[1:]

            if len(pred_params) == 0:
                true_relevant_facts.append(pred)
            else:
                action_objects = True

                for param in pred_params:
                    if param not in a_params:
                        action_objects = False
                        break

                if action_objects:
                    true_relevant_facts.append(pred)

        return true_relevant_facts


    def get_action_positive_preconditions(self, action_label):

        action_op = action_label.strip().split("(")[0]
        op_precs = sorted(self.get_operator_preconditions(action_op))

        action_params = action_label.strip()[:-1].split("(")[1].strip().split(",")

        all_op_precs = "++".join(op_precs)

        for i in range(len(action_params)):
            obj = action_params[i]

            all_op_precs = all_op_precs.replace("?param_{} ".format(i+1),"{} ".format(obj))
            all_op_precs = all_op_precs.replace("?param_{})".format(i+1),"{})".format(obj))

        return all_op_precs.split("++")



    def get_op_positive_precondition(self, action_label):

        operator = action_label.strip().split("(")[0]
        a_params = action_label[action_label.find("(") + 1:action_label.find(")")].split(",")
        # negative_preconditions = self.get_true_relevant_predicates(action_label)
        positive_preconditions = self.get_true_relevant_predicates(action_label)
        positive_preconditions_placeholder = []
        params_placeholder = defaultdict(list)
        checked_params = []
        for el in a_params:

            if el not in checked_params:
                el_indexes = [i for i in range(len(a_params)) if a_params[i] == el]

                for index in el_indexes:
                    params_placeholder[el].append("?param_{}".format(index+1))

                checked_params.append(el)

        for pred in positive_preconditions:
            pred_params = pred[1:-1].split()[1:]

            pred_params_placeholder = [params_placeholder[obj] for obj in pred_params]

            var_params_combinations = [list(p) for p in itertools.product(*pred_params_placeholder)]

            for tup in var_params_combinations:

                newpred = pred

                for k, subst in enumerate(tup):

                    if newpred.find(pred_params[k] + " ") != -1:
                        newpred = newpred.replace(pred_params[k] + " ", subst + " ", 1)
                    elif newpred.find(pred_params[k] + ")") != -1:
                        newpred = newpred.replace(pred_params[k] + ")", subst + ")", 1)
                    else:
                        print("Something went wrong when replacing objects with parameters placeholder "
                              "in get_negative_precondition()")

                positive_preconditions_placeholder.append(newpred)

        return positive_preconditions_placeholder


    def add_negative_precondition_fixed(self, action_label, negative_precond):

        operator = action_label.strip().split("(")[0]

        # Check redundant negative preconditions
        for precond in self.operator_negative_preconditions[operator]:
            if set(negative_precond).issubset(set(precond)):
                self.operator_negative_preconditions[operator].remove(precond)

        self.operator_negative_preconditions[operator].append(negative_precond)


    def compute_not_executable_actionsJAVA(self):
        """
        Compute not executable action list with a java program
        :return: None
        """


        with open("PDDL/facts.pddl", "r") as f:
            data = [el.strip() for el in f.read().split("\n")]
            facts = re.findall(":init.*\(:goal","".join(data))[0]
        current_state_pddl = sorted(re.findall("\([^()]*\)", facts))

        # Write current pddl state into a text file
        with open(os.path.join("Info","current_state.txt"), "w") as f:
            [f.write(el + "\n") for el in current_state_pddl]

        # Write all action list into a text file

        # if not os.path.exists(os.path.join("Info","action_list.txt")):
        with open(os.path.join("Info","action_list.txt"), "w") as file:
            [file.write("{} ++ {}\n".format(self.action_labels[i], i)) for i in range(len(self.action_labels))]

        # Write action negative preconditions into a json file
        with open(os.path.join("Info","action_executability_constraints.json"), "w") as outfile:
            # json.dump(self.operator_negative_preconditions, outfile)
            json.dump(self.operator_executability_constr, outfile, indent=2)

        if len(self.not_executable_actions_index) == 0:
            bash_command = "{} -jar compute_not_executable_actions.jar".format(Configuration.JAVA_BIN_PATH)
        else:
            bash_command = "{} -jar compute_not_executable_actions.jar {}".format(Configuration.JAVA_BIN_PATH,
                                                       self.last_failed_action)

        # process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

        subprocess.call(bash_command.split(), stderr=subprocess.PIPE)

        with open(os.path.join("Info", "not_executable_action_list.txt"), "r") as f:
            lines = [el for el in f.read().split("\n") if el.strip() != ""]
            not_executable_actions = list(map(int, lines))

        return not_executable_actions


    def eval_logOLD(self):
        """
        Evaluate metrics and print them in log file
        :return: None
        """

        preconditions_recall = metrics.action_model_prec_recall()
        preconditions_precision = metrics.action_model_prec_precision()

        effects_recall = metrics.action_model_eff_recall()
        effects_precision = metrics.action_model_eff_precision()

        overall_recall = metrics.action_model_overall_recall()
        overall_precision = metrics.action_model_overall_precision()

        evaluate = {'timestamp': "{0:.2f}".format(round(sum(self.time_at_iter), 2), 2),
                    'iter': self.iter,
                    # 'nr_of_states': self.model.nr_of_states,
                    # 'coverage': metrics.coverage(self.model, simulator),
                    'Preconditions recall': "{0:.2f}".format(preconditions_recall),
                    'Preconditions precision': "{0:.2f}".format(preconditions_precision),
                    'Effects recall': "{0:.2f}".format(effects_recall),
                    'Effects precision': "{0:.2f}".format(effects_precision),
                    'Overall recall': "{0:.2f}".format(overall_recall),
                    'Overall precision': "{0:.2f}".format(overall_precision)
                    }
        print("\t\t".join([str(i) for i in list(evaluate.values())]))
        self.eval = self.eval.append(evaluate, ignore_index=True)

        # DEBUG MODEL MEMORY CONSUMPTION
        # print(asizeof.asized(self.model.__dict__, detail=1).format())


    def eval_log(self):
        """
        Evaluate metrics and print them in log file
        :return: None
        """

        real_precs_size, learned_precs_size, real_eff_pos_size, learned_eff_pos_size, \
        real_eff_neg_size, learned_eff_neg_size, ins_pre, del_pre, ins_eff_pos, del_eff_pos, \
        ins_eff_neg, del_eff_neg, precs_recall, eff_pos_recall, eff_neg_recall, precs_precision, \
        eff_pos_precision, eff_neg_precision, overall_recall, overall_precision = metrics.action_model_statistics()


        evaluate = {'Time': "{0:.2f}".format(round(sum(self.time_at_iter), 2), 2),
                    'Iter': self.iter,
                    'Real_precs': int(real_precs_size),
                    'Learn_precs': int(learned_precs_size),
                    'Real_pos': int(real_eff_pos_size),
                    'Learn_pos': int(learned_eff_pos_size),
                    'Real_neg': int(real_eff_neg_size),
                    'Learn_neg': int(learned_eff_neg_size),
                    'Ins_pre': int(ins_pre),
                    'Del_pre': int(del_pre),
                    'Ins_pos': int(ins_eff_pos),
                    'Del_pos': int(del_eff_pos),
                    'Ins_neg': int(ins_eff_neg),
                    'Del_neg': int(del_eff_neg),
                    'Precs_recall': "{0:.2f}".format(precs_recall),
                    'Pos_recall': "{0:.2f}".format(eff_pos_recall),
                    'Neg_recall': "{0:.2f}".format(eff_neg_recall),
                    'Precs_precision': "{0:.2f}".format(precs_precision),
                    'Pos_precision': "{0:.2f}".format(eff_pos_precision),
                    'Neg_precision': "{0:.2f}".format(eff_neg_precision),
                    'Tot_recall': "{0:.2f}".format(overall_recall),
                    'Tot_precision': "{0:.2f}".format(overall_precision)
                    }
        # print("\t\t".join([str(i) for i in list(evaluate.values())]))
        self.eval = self.eval.append(evaluate, ignore_index=True)

        print("\n")
        print(template.format(  # header
            Time="Time", Iter="Iter", Real_precs="Real_precs", Learn_precs="Learn_precs",
            Real_pos="Real_pos", Learn_pos="Learn_pos", Real_neg="Real_neg", Learn_neg="Learn_neg",
            Ins_pre="Ins_pre", Del_pre="Del_pre", Ins_pos="Ins_pos", Del_pos="Del_pos",
            Ins_neg="Ins_neg", Del_neg="Del_neg", Precs_recall="Precs_recall",
            Pos_recall="Pos_recall", Neg_recall="Neg_recall", Precs_precision="Precs_precision",
            Pos_precision="Pos_precision", Neg_precision="Neg_precision",
            Tot_recall="Tot_recall", Tot_precision="Tot_precision"
        ))
        print(template.format(**evaluate))
        print("\n")

        # DEBUG MODEL MEMORY CONSUMPTION
        # print(asizeof.asized(self.model.__dict__, detail=1).format())


    def eval_log_with_uncertain_neg(self):
        """
        Evaluate metrics and print them in log file
        :return: None
        """

        real_precs_size, learned_precs_size, real_eff_pos_size, learned_eff_pos_size, \
        real_eff_neg_size, learned_eff_neg_size, ins_pre, del_pre, ins_eff_pos, del_eff_pos, \
        ins_eff_neg, del_eff_neg, precs_recall, eff_pos_recall, eff_neg_recall, precs_precision, \
        eff_pos_precision, eff_neg_precision, overall_recall, overall_precision = metrics.action_model_statistics_with_uncertain_neg(self.uncertain_negative_effects)


        evaluate = {'Time': "{0:.2f}".format(round(sum(self.time_at_iter), 2), 2),
                    'Iter': self.iter,
                    'Real_precs': int(real_precs_size),
                    'Learn_precs': int(learned_precs_size),
                    'Real_pos': int(real_eff_pos_size),
                    'Learn_pos': int(learned_eff_pos_size),
                    'Real_neg': int(real_eff_neg_size),
                    'Learn_neg': int(learned_eff_neg_size),
                    'Ins_pre': int(ins_pre),
                    'Del_pre': int(del_pre),
                    'Ins_pos': int(ins_eff_pos),
                    'Del_pos': int(del_eff_pos),
                    'Ins_neg': int(ins_eff_neg),
                    'Del_neg': int(del_eff_neg),
                    'Precs_recall': "{0:.2f}".format(precs_recall),
                    'Pos_recall': "{0:.2f}".format(eff_pos_recall),
                    'Neg_recall': "{0:.2f}".format(eff_neg_recall),
                    'Precs_precision': "{0:.2f}".format(precs_precision),
                    'Pos_precision': "{0:.2f}".format(eff_pos_precision),
                    'Neg_precision': "{0:.2f}".format(eff_neg_precision),
                    'Tot_recall': "{0:.2f}".format(overall_recall),
                    'Tot_precision': "{0:.2f}".format(overall_precision)
                    }
        # print("\t\t".join([str(i) for i in list(evaluate.values())]))
        self.eval = self.eval.append(evaluate, ignore_index=True)

        print("\n")
        print(template.format(  # header
            Time="Time", Iter="Iter", Real_precs="Real_precs", Learn_precs="Learn_precs",
            Real_pos="Real_pos", Learn_pos="Learn_pos", Real_neg="Real_neg", Learn_neg="Learn_neg",
            Ins_pre="Ins_pre", Del_pre="Del_pre", Ins_pos="Ins_pos", Del_pos="Del_pos",
            Ins_neg="Ins_neg", Del_neg="Del_neg", Precs_recall="Precs_recall",
            Pos_recall="Pos_recall", Neg_recall="Neg_recall", Precs_precision="Precs_precision",
            Pos_precision="Pos_precision", Neg_precision="Neg_precision",
            Tot_recall="Tot_recall", Tot_precision="Tot_precision"
        ))
        print(template.format(**evaluate))
        print("\n")

        # DEBUG MODEL MEMORY CONSUMPTION
        # print(asizeof.asized(self.model.__dict__, detail=1).format())


    # Add conjunction of action variable predicates
    def add_operator_precondition(self, action_label):

        a_name = action_label.split("(")[0]
        a_params = action_label[
                   action_label.find("(") + 1:action_label.find(")")].split(",")

        # # Check if action contains equal objects
        # if len(a_params) != len(set(a_params)):
        #     return False

        # Get current pddl state
        with open("PDDL/facts.pddl", "r") as f:
            data = [el.strip() for el in f.read().split("\n")]
            facts = re.findall(":init.*\(:goal","".join(data))[0]

        current_state_pddl = sorted(re.findall("\([^()]*\)", facts))

        # Read current action model
        with open("PDDL/domain_learned.pddl", 'r') as f:
            data = f.read().split("\n")

        # Update current action model with a new operator precondition
        with open("PDDL/domain_learned.pddl", 'w') as f:

            action_model_updated = False

            action_separator_char = None
            check_syntax = False
            for i in range(2, len(data)):

                if data[i-2].strip().find('(:action') != -1 and not check_syntax:
                    if data[i-2].strip().split()[1].find("-") != -1:
                        action_separator_char = "-"
                        check_syntax = True

                    elif data[i - 2].strip().split()[1].find("_") != -1:
                        action_separator_char = "_"
                        check_syntax = True

                if check_syntax:
                    a_name_checked = a_name.lower().replace('_', action_separator_char).replace('-', action_separator_char)
                else:
                    a_name_checked  = a_name.lower()

                if data[i-2].lower().strip().find("(:action " + a_name_checked) != -1:

                    relevant_facts = []
                    for el in current_state_pddl:
                        found = True
                        for var in el.strip()[1:-1].split()[1:]:
                            if var not in a_params:
                                found = False
                                break
                        if found:
                            new_precond = el.strip()
                            precond_params = new_precond[1:-1].split()[1:]

                            if len(precond_params) > 0:
                                var_action_parameters_indexes = []
                                checked_vars = []
                                for var in precond_params:

                                    # if var not in checked_vars:
                                    var_action_indexes = []
                                    for k in range(len(a_params)):
                                        if a_params[k] == var:
                                            var_action_indexes.append(k)

                                    var_action_parameters_indexes.append(var_action_indexes)
                                    checked_vars.append(var)
                                    #

                                var_params_combinations = [list(p) for p in itertools.product(*var_action_parameters_indexes)]

                                # Add a precondition predicate for each possible parameter combination
                                for tup in var_params_combinations:

                                    tmp_new_precond = new_precond

                                    for m,var_index in enumerate(tup):

                                        if tmp_new_precond.find(precond_params[m] + " ") != -1:
                                            tmp_new_precond = tmp_new_precond\
                                                .replace(precond_params[m] + " ", "?param_{} ".format(var_index + 1), 1)
                                        else:
                                            tmp_new_precond = tmp_new_precond\
                                            .replace(precond_params[m] + ")", "?param_{})".format(var_index + 1), 1)

                                    # Add predicate to operator preconditions
                                    relevant_facts.append(tmp_new_precond)

                            # If the predicate does not contain any parameter
                            else:
                                # Add predicate to operator preconditions
                                relevant_facts.append(new_precond)

                    # relevant_facts.extend(self.operator_certain_predicates[a_name.upper()])
                    relevant_facts.extend(self.operator_certain_predicates[a_name])

                    relevant_facts = list(set(relevant_facts))

                    if data[i+1].strip() != " ".join(sorted(relevant_facts)).strip():
                        if data[i+1].strip() == ")":
                            data[i] = data[i] + "\n\t\t\t\t\t{}".format(" ".join(sorted(relevant_facts)))
                            action_model_updated = True
                            self.operator_uncertain_predicates[a_name] = relevant_facts

                            self.update_executability_constr(a_name, relevant_facts)

                            # Add negative preconditions to filter action list
                            # self.add_negative_precondition_fixed(action_label, relevant_facts)

                            # Keep only preconditions in possible negative effects
                            # self.possible_negative_effects[a_name] = [pred for pred in self.possible_negative_effects[a_name]
                            #                                           if pred in relevant_facts]

                        else:
                            current_precondition = sorted(["(" + el.strip()[:] for el in data[i+1].strip().split("(") if el.strip() != ""])

                            precond_intersection = [el for el in relevant_facts if el in current_precondition]
                            data[i+1] = "\t\t\t\t\t{}".format(" ".join(sorted(precond_intersection)))

                            if len(precond_intersection) != len(current_precondition):
                                action_model_updated = True

                                # DEBUG
                                print("operator {}, removed uncertain preconditions {}".format(a_name_checked,
                                                                                     [el for el in current_precondition
                                                                                      if el not in precond_intersection]))

                                self.operator_uncertain_predicates[a_name] = []
                                for el in precond_intersection:
                                    if el not in self.operator_certain_predicates[a_name]:
                                        self.operator_uncertain_predicates[a_name].append(el)

                                # self.add_negative_precondition_fixed(action_label, precond_intersection)

                                self.update_executability_constr(a_name, precond_intersection)

                                # Remove deleted preconditions from possible negative effects
                                # self.possible_negative_effects[a_name] = [pred for pred in self.possible_negative_effects[a_name]
                                #                                           if pred in precond_intersection]


                    break

            [f.write(el + "\n") for el in data]

            if a_name not in self.operator_learned:
                self.operator_learned.append(a_name)

        # Update domain dummy (i.e., domain with only learned operators)
        shutil.copyfile("PDDL/domain_learned.pddl", os.path.join("PDDL/domain_dummy.pddl"))
        with open("PDDL/domain_dummy.pddl", 'r') as f:
            data = [el for el in f.read().split("\n") if el.strip() != ""]

            removed_rows = []

            for m in range(len(data)):

                row = data[m]

                if row.lower().strip().startswith("(:action"):

                    if data[m + 2].strip().startswith(":precondition") and data[m + 3].strip() == ")":
                        removed_rows.append(m)

                        for n in range(m + 1, len(data) - 1):
                            if data[n].strip().startswith("(:action"):
                                break
                            removed_rows.append(n)

            dummy_domain = [data[i] for i in range(len(data)) if i not in removed_rows]

        with open("PDDL/domain_dummy.pddl", 'w') as f:
            [f.write(el + "\n") for el in dummy_domain]

        return action_model_updated


    # Add conjunction of action variable predicates
    def add_all_operator_precs(self, op):

        op_precs = self.get_op_relevant_predicates(op)

        a_name_checked = op

        # Read current action model
        with open("PDDL/domain_learned.pddl", 'r') as f:
            data = f.read().split("\n")

        # Update current action model with a new operator precondition
        with open("PDDL/domain_learned.pddl", 'w') as f:

            for i in range(2, len(data)):

                if data[i-2].lower().strip().find("(:action " + a_name_checked) != -1:

                    relevant_facts = op_precs

                    if data[i+1].strip() == ")":
                        data[i] = data[i] + "\n\t\t\t\t\t{}".format(" ".join(sorted(relevant_facts)))
                        break
                    else:
                        print('ERROR: check the method in Learner.py which add preconditions of never executed operators')
                        exit()

            [f.write(el + "\n") for el in data]



    # Add conjunction of action variable predicates
    def add_operator_effects(self, action_label, old_state, current_state_pddl):

        effects_updated = False

        a_name = action_label.split("(")[0]
        a_params = action_label[
                   action_label.find("(") + 1:action_label.find(")")].split(",")

        # # Check if action contains equal input objects
        # if len(a_params) != len(set(a_params)):
        #     return False

        # Read current action model
        with open("PDDL/domain_learned.pddl", 'r') as f:
            data = f.read().split("\n")

        # Update current action model with some new operator effects
        all_possible_effects = self.get_op_relevant_predicates(a_name)
        with open("PDDL/domain_learned.pddl", 'w') as f:

            action_model_updated = False

            for i in range(2, len(data)):

                if data[i-2].lower().strip().find("(:action " + a_name) != -1\
                        and len(data[i-2].lower().strip().replace("(:action " + a_name, "")) == 0:

                    positive_effects_obj = [el for el in current_state_pddl if el not in old_state]
                    negative_effects_obj = [el for el in old_state if el not in current_state_pddl]








############################ QUESTA PARTE è STATA AGGIUNTA DOPO I TEST, SE CI SONO PROBLEMI
############################ RIMUOVERLA E SCOMMENTARE LE TRE RIGHE A INIZIO METODO


                    # Remove ambiguous effects
                    pos_removed = []

                    for pos in positive_effects_obj:
                        correctness = True
                        eff_params = pos[1:-1].split()[1:]
                        for obj in eff_params:
                            if sum([obj.strip() == a_obj for a_obj in a_params]) > 1:
                                correctness = False
                        if not correctness:
                            pos_removed.append(pos)

                    neg_removed = []

                    for neg in negative_effects_obj:
                        correctness = True
                        eff_params = neg[1:-1].split()[1:]
                        for obj in eff_params:
                            if sum([obj.strip() == a_obj for a_obj in a_params]) > 1:
                                correctness = False
                        if not correctness:
                            neg_removed.append(neg)

                    positive_effects_obj = [pos for pos in positive_effects_obj
                                            if pos not in pos_removed]
                    negative_effects_obj = [neg for neg in negative_effects_obj
                                            if neg not in neg_removed]
###################################################################################




                    positive_effects = []
                    negative_effects = []

                    # Lift positive effects to operator parameters
                    for eff in positive_effects_obj:

                            new_pos_eff = eff.strip()
                            pos_eff_params = new_pos_eff[1:-1].split()[1:]

                            if len(pos_eff_params) > 0:
                                var_action_parameters_indexes = []
                                checked_vars = []

                                # duplicates_input_objs = False

                                for var in pos_eff_params:

                                    # if var not in checked_vars:
                                    var_action_indexes = []
                                    for k in range(len(a_params)):
                                        if a_params[k] == var:
                                            var_action_indexes.append(k)

                                    var_action_parameters_indexes.append(var_action_indexes)
                                        # checked_vars.append(var)

                                    # if len(var_action_indexes) > 1:
                                    #     duplicates_input_objs = True
                                    #     break

                                var_params_combinations = [list(p) for p in itertools.product(*var_action_parameters_indexes)]

                                # if not duplicates_input_objs:
                                # Add a effect predicate for each possible parameter combination
                                for tup in var_params_combinations:

                                    tmp_new_pos_eff = new_pos_eff

                                    for m,var_index in enumerate(tup):

                                        if tmp_new_pos_eff.find("{} ".format(pos_eff_params[m])) != -1:
                                            tmp_new_pos_eff = tmp_new_pos_eff\
                                            .replace(pos_eff_params[m] + " ", "?param_{} ".format(var_index + 1), 1)
                                        else:
                                            tmp_new_pos_eff = tmp_new_pos_eff\
                                            .replace(pos_eff_params[m] + ")", "?param_{})".format(var_index + 1), 1)

                                        # tmp_new_pos_eff = tmp_new_pos_eff\
                                        #     .replace(pos_eff_params[m] + " ", "?param_{} ".format(var_index + 1))\
                                        #     .replace(pos_eff_params[m] + ")", "?param_{})".format(var_index + 1))

                                    # Add predicate to operator positive effects
                                    positive_effects.append(tmp_new_pos_eff)

                            # If the predicate does not contain any parameter
                            else:
                                # Add predicate to operator positive effects
                                positive_effects.append(new_pos_eff)


                    # Lift negative effects to operator parameters
                    for eff in negative_effects_obj:

                            new_neg_eff = eff.strip()
                            neg_eff_params = new_neg_eff[1:-1].split()[1:]

                            if len(neg_eff_params) > 0:
                                var_action_parameters_indexes = []
                                # checked_vars = []
                                # duplicates_input_objs = False

                                for var in neg_eff_params:

                                    # if var not in checked_vars:
                                    var_action_indexes = []
                                    for k in range(len(a_params)):
                                        if a_params[k] == var:
                                            var_action_indexes.append(k)

                                    var_action_parameters_indexes.append(var_action_indexes)

                                    # if len(var_action_indexes) > 1:
                                    #     duplicates_input_objs = True
                                    #     break

                                var_params_combinations = [list(p) for p in itertools.product(*var_action_parameters_indexes)]

                                # if not duplicates_input_objs:
                                    # Add a effect predicate for each possible parameter combination
                                for tup in var_params_combinations:

                                    tmp_new_neg_eff = new_neg_eff

                                    for m, var_index in enumerate(tup):

                                        if tmp_new_neg_eff.find("{} ".format(neg_eff_params[m])) != -1:
                                            tmp_new_neg_eff = tmp_new_neg_eff \
                                                .replace(neg_eff_params[m] + " ", "?param_{} ".format(var_index + 1), 1)
                                        else:
                                            tmp_new_neg_eff = tmp_new_neg_eff \
                                                .replace(neg_eff_params[m] + ")", "?param_{})".format(var_index + 1), 1)

                                    # Add predicate to operator preconditions
                                    negative_effects.append("(not " + tmp_new_neg_eff + ")")

                            # If the predicate does not contain any parameter
                            else:
                                # Add predicate to operator positive effects
                                negative_effects.append("(not " + new_neg_eff + ")")


                    # Check if there is an evidence of a certain positive or negative effect (this is required because of
                    # the strip syntactic assumption that all negative effects must be positive preconditions. If
                    # there is an evidence of a certain negative effect, it is added to the action schema even if
                    # it is not a precondition.
                    for pos in positive_effects_obj:
                        eff_params = pos.strip()[1:-1].split()[1:]
                        correctness = True
                        for obj in eff_params:
                            if sum([obj == a_obj for a_obj in a_params]) > 1:
                                correctness = False
                        if correctness:

                            for v, param in enumerate(a_params):
                                pos = pos.replace(" {} ".format(a_params[v]), " ?param_{} ".format(v+1))
                                pos = pos.replace(" {})".format(a_params[v]), " ?param_{})".format(v+1))

                            if pos not in self.certain_positive_effects[a_name]:
                                print("Operator {}, adding certain positive effect {}".format(a_name, pos))
                                self.certain_positive_effects[a_name].append(pos)

                                if pos in self.uncertain_positive_effects[a_name]:
                                    self.uncertain_positive_effects[a_name].remove(pos)


                    for neg in negative_effects_obj:
                        eff_params = neg.strip()[1:-1].split()[1:]
                        correctness = True
                        for obj in eff_params:
                            if sum([obj == a_obj for a_obj in a_params]) > 1:
                                correctness = False
                        if correctness:

                            for v, param in enumerate(a_params):
                                neg = neg.replace(" {} ".format(a_params[v]), " ?param_{} ".format(v+1))
                                neg = neg.replace(" {})".format(a_params[v]), " ?param_{})".format(v+1))

                            # neg = "(not {})".format(neg)

                            if "(not {})".format(neg) not in self.certain_negative_effects[a_name]:
                                print("Operator {}, adding certain negative effect {}".format(a_name, "(not {})".format(neg)))
                                self.certain_negative_effects[a_name].append("(not {})".format(neg))

                                if neg in self.uncertain_negative_effects[a_name]:
                                    self.uncertain_negative_effects[a_name].remove(neg)





                    # Compute surely not positive effects, i.e., possible effects which are false
                    # in the current state obtained after executing the action
                    all_possible_effects_obj = copy.deepcopy(self.uncertain_positive_effects[a_name])
                    all_possible_effects = copy.deepcopy(self.uncertain_positive_effects[a_name])
                    for k, pred in enumerate(all_possible_effects_obj):

                        for v, param in enumerate(a_params):

                            all_possible_effects_obj[k] = all_possible_effects_obj[k].replace(" ?param_{} ".format(v+1),
                                                                                              " {} ".format(a_params[v]))
                            all_possible_effects_obj[k] = all_possible_effects_obj[k].replace(" ?param_{})".format(v+1),
                                                                                              " {})".format(a_params[v]))
                    for k, fact in enumerate(all_possible_effects_obj):
                        # if fact not in old_state and fact not in current_state_pddl:
                        if fact not in current_state_pddl:
                            if all_possible_effects[k] in self.uncertain_positive_effects[a_name]:
                                # print("Operator {}, removed possible positive effect {}".format(a_name, all_possible_effects[k]))

                                print("removing {} from possible positive effects".format(all_possible_effects[k]))
                                self.uncertain_positive_effects[a_name].remove(all_possible_effects[k])


                    # Compute surely not negative effects, i.e., possible effects which are
                    # true in the current state obtained after executing the action
                    all_possible_effects_obj = copy.deepcopy(self.uncertain_negative_effects[a_name])
                    all_possible_effects = copy.deepcopy(self.uncertain_negative_effects[a_name])
                    for k, pred in enumerate(all_possible_effects_obj):

                        for v, param in enumerate(a_params):

                            all_possible_effects_obj[k] = all_possible_effects_obj[k].replace(" ?param_{} ".format(v+1),
                                                                                              " {} ".format(a_params[v]))
                            all_possible_effects_obj[k] = all_possible_effects_obj[k].replace(" ?param_{})".format(v+1),
                                                                                              " {})".format(a_params[v]))
                    for k, fact in enumerate(all_possible_effects_obj):
                        # if fact in old_state and fact in current_state_pddl:
                        if fact in current_state_pddl:
                            # if all_possible_effects[k] not in self.possible_positive_effects[a_name]:

                            # eff_params = all_possible_effects_obj[k].strip()[1:-1].split()[1:]
                            #
                            # # If an effect object appears more than once in action objects, then
                            # # it could be that a negative effect is not correctly fixed as impossible negative effect
                            # correctness = True
                            # for obj in eff_params:
                            #     if sum([obj == a_obj for a_obj in a_params]) > 1:
                            #         correctness = False
                            # if correctness:
                            #     # print("Operator {}, removed possible negative effect {}".format(a_name, all_possible_effects[k]))
                            #     self.possible_negative_effects[a_name].remove(all_possible_effects[k])



                            # If an effect object appears more than once in action objects, then
                            # it could be that a negative effect is not correctly fixed as impossible negative effect
                            eff_params = all_possible_effects_obj[k].strip()[1:-1].split()[1:]
                            correctness = True
                            for obj in eff_params:
                                if sum([obj == a_obj for a_obj in a_params]) > 1:
                                    correctness = False
                            if correctness:
                                print("removing {} from possible negative effects".format(all_possible_effects[k]))
                                self.uncertain_negative_effects[a_name].remove(all_possible_effects[k])


                    # Remove surely not positive effects from new positive effects
                    # [positive_effects.remove(e) for e in self.uncertain_positive_effects[a_name] if e in positive_effects]
                    # [positive_effects.remove(e) for e in positive_effects if e not in self.possible_positive_effects[a_name]]

                    # Add only new positive effects
                    positive_effects = [el for el in positive_effects if el in self.uncertain_positive_effects[a_name]]

                    # Remove surely not negative effects from new negative effects
                    # [negative_effects.remove("(not " + e + ")") for e in self.uncertain_negative_effects[a_name]
                    #  if "(not " + e + ")" in negative_effects]
                    # [negative_effects.remove(e) for e in negative_effects
                    #  if e.replace("(not","").strip()[:-1] not in self.possible_negative_effects[a_name]]

                    # Add only new negative effects
                    negative_effects = [el for el in negative_effects
                                        if el.replace("(not","").strip()[:-1] in self.uncertain_negative_effects[a_name]]

                    for j in range(i, len(data)):

                        if data[j].find(":effect") != -1:

                            cur_neg_effect = re.findall("\(not[^)]*\)\)", data[j])
                            cur_pos_effect = [el for el in re.findall("\([^()]*\)", data[j])
                                              if el not in [el.replace("(not","").strip()[:-1] for el in cur_neg_effect]
                                              and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

                            all_pos_eff = list(set(positive_effects + cur_pos_effect))
                            all_neg_eff = list(set(negative_effects + cur_neg_effect))


                            uncertain_pos_eff = [e for e in cur_pos_effect if e not in positive_effects]
                            uncertain_neg_eff = [e for e in cur_neg_effect if e not in negative_effects]
                            uncertain_pos_eff_obj = copy.deepcopy(uncertain_pos_eff)
                            uncertain_neg_eff_obj = copy.deepcopy(uncertain_neg_eff)

                            for k in range(len(uncertain_pos_eff_obj)):
                                for v in range(len(a_params)):
                                    uncertain_pos_eff_obj[k] = uncertain_pos_eff_obj[k].replace(" ?param_{} ".format(v+1), " {} ".format(a_params[v]))
                                    uncertain_pos_eff_obj[k] = uncertain_pos_eff_obj[k].replace(" ?param_{})".format(v+1), " {})".format(a_params[v]))

                            for k in range(len(uncertain_neg_eff_obj)):
                                for v in range(len(a_params)):
                                    uncertain_neg_eff_obj[k] = uncertain_neg_eff_obj[k].replace(" ?param_{} ".format(v+1), " {} ".format(a_params[v]))
                                    uncertain_neg_eff_obj[k] = uncertain_neg_eff_obj[k].replace(" ?param_{})".format(v+1), " {})".format(a_params[v]))


                            # Detect inconsistencies among positive and negative effects, if an inconsistency occur,
                            # the agent does not consider such effects, since the incosistency is due to equal objects
                            # in input for a particular action (and generally not for all actions)

                            to_remove = [eff for eff in uncertain_pos_eff_obj if "(not {})".format(eff) in uncertain_neg_eff_obj]


                            uncertain_pos_eff_obj = [eff for eff in uncertain_pos_eff_obj if eff not in to_remove]
                            uncertain_neg_eff_obj = [eff for eff in uncertain_neg_eff_obj if "(not {})".format(eff) not in to_remove]

                            # pos_eff_removed = [uncertain_pos_eff[k] for k, e in enumerate(uncertain_pos_eff_obj)
                            #                    if e not in old_state]
                            # neg_eff_removed = [uncertain_neg_eff[k] for k, e in enumerate(uncertain_neg_eff_obj)
                            #                    if e.strip().replace("(not", "").strip()[:-1] in old_state
                            #                    and e.strip().replace("(not", "").strip()[:-1] in current_state_pddl]

                            pos_eff_removed = [uncertain_pos_eff[k] for k, e in enumerate(uncertain_pos_eff_obj)
                                               if e not in current_state_pddl]
                            neg_eff_removed = [uncertain_neg_eff[k] for k, e in enumerate(uncertain_neg_eff_obj)
                                               if e.strip().replace("(not", "").strip()[:-1] in current_state_pddl]

                            # [all_pos_eff.remove(e) for e in pos_eff_removed]
                            # [all_neg_eff.remove(e) for e in neg_eff_removed]
                            all_pos_eff = [eff for eff in all_pos_eff if eff not in pos_eff_removed]
                            all_neg_eff = [eff for eff in all_neg_eff if eff not in neg_eff_removed]

                            # # Check inconsistent effects
                            # inconsistent_effects = [eff for eff in all_pos_eff if "(not {})".format(eff) in all_neg_eff]
                            # for incons_eff in inconsistent_effects:
                            #     print("Operator {}, removing inconsistent effect {}".format(a_name, incons_eff))
                            #     all_pos_eff.remove(incons_eff)
                            #     all_neg_eff.remove("(not {})".format(incons_eff))
                            #
                            #     if incons_eff in self.possible_positive_effects[a_name]:
                            #         self.possible_positive_effects[a_name].remove(incons_eff)
                            #         effects_removed = True
                            #     if incons_eff in self.possible_negative_effects[a_name]:
                            #         self.possible_negative_effects[a_name].remove(incons_eff)
                            #         effects_removed = True

                            all_pos_eff = list(set(all_pos_eff + self.certain_positive_effects[a_name]))
                            all_neg_eff = list(set(all_neg_eff + self.certain_negative_effects[a_name]))



                            # Detect inconsistent effects
                            inconsistent_eff = []
                            for pos in all_pos_eff:
                                if "(not {})".format(pos) in all_neg_eff:
                                    inconsistent_eff.append(pos)

                            if len(inconsistent_eff) > 0:
                                effects_updated = True
                                all_pos_eff = [pos for pos in all_pos_eff if pos not in inconsistent_eff]
                                all_neg_eff = [neg for neg in all_neg_eff if neg.replace("(not","").strip()[:-1] not in inconsistent_eff]

                                self.uncertain_positive_effects[a_name] = [pos for pos in self.uncertain_positive_effects[a_name]
                                                                           if pos not in inconsistent_eff]
                                self.uncertain_negative_effects[a_name] = [neg for neg in self.uncertain_negative_effects[a_name]
                                                                           if neg not in inconsistent_eff]



                            data[j] = ":effect (and {} {}))".format(" ".join(all_pos_eff), " ".join(all_neg_eff))

                            # Check if some previous effect has been removed
                            if not effects_updated:
                                effects_updated = (sum([eff not in all_pos_eff for eff in cur_pos_effect]) \
                                                  + sum([eff not in all_neg_eff for eff in cur_neg_effect])) > 0 \
                                                  or (sum([eff not in cur_pos_effect for eff in all_pos_eff]) \
                                                  + sum([eff not in cur_neg_effect for eff in all_neg_eff])) > 0


                            # Check for new certain effects
                            all_pos_eff_obj = copy.deepcopy(all_pos_eff)
                            all_neg_eff_obj = copy.deepcopy(all_neg_eff)
                            all_neg_eff_obj = [el.replace("(not","").strip()[:-1].strip() for el in all_neg_eff_obj]
                            # all_pos_eff_obj = copy.deepcopy(positive_effects)
                            # all_neg_eff_obj = copy.deepcopy(negative_effects)
                            # all_neg_eff_obj = [el.replace("(not","").strip()[:-1].strip() for el in all_neg_eff_obj]

                            for u in range(len(all_pos_eff_obj)):
                                for k in range(len(a_params)):
                                    all_pos_eff_obj[u] = all_pos_eff_obj[u].replace(" ?param_{} ".format(k+1), " {} ".format(a_params[k]))
                                    all_pos_eff_obj[u] = all_pos_eff_obj[u].replace(" ?param_{})".format(k+1), " {})".format(a_params[k]))

                            for u in range(len(all_neg_eff_obj)):
                                for k in range(len(a_params)):
                                    all_neg_eff_obj[u] = all_neg_eff_obj[u].replace(" ?param_{} ".format(k+1), " {} ".format(a_params[k]))
                                    all_neg_eff_obj[u] = all_neg_eff_obj[u].replace(" ?param_{})".format(k+1), " {})".format(a_params[k]))

                            # Detect certain positive effects:
                            for u, eff in enumerate(all_pos_eff_obj):

                                # # If there are no inconsistencies with the negative effects, and the positive
                                # # effect appears only once in all positive effects, then it is a certain one
                                # if eff not in all_neg_eff_obj and sum([eff == eff2 for eff2 in all_pos_eff_obj]) == 1\
                                #     and positive_effects[u] not in self.certain_positive_effects[a_name]:
                                #     print("Operator {}, adding certain positive effect {}".format(a_name, positive_effects[u]))
                                #     self.certain_positive_effects[a_name].append(positive_effects[u])
                                #
                                #     if positive_effects[u] in self.possible_positive_effects[a_name]:
                                #         self.possible_positive_effects[a_name].remove(positive_effects[u])
                                #
                                #     if positive_effects[u] in self.possible_negative_effects[a_name]:
                                #         self.possible_negative_effects[a_name].remove(positive_effects[u])

                                # If there are no inconsistencies with the negative effects, and the positive
                                # effect appears only once in all positive effects, then it is a certain one
                                if eff not in all_neg_eff_obj and sum([eff == eff2 for eff2 in all_pos_eff_obj]) == 1\
                                    and all_pos_eff[u] not in self.certain_positive_effects[a_name]\
                                        and (eff not in old_state and eff in current_state_pddl):

                                    eff_params = eff.strip()[1:-1].split()[1:]
                                    correctness = True
                                    for obj in eff_params:
                                        if sum([obj == a_obj for a_obj in a_params]) > 1:
                                            correctness = False

                                    if correctness:
                                        print("Operator {}, adding certain positive effect {}".format(a_name, all_pos_eff[u]))
                                        self.certain_positive_effects[a_name].append(all_pos_eff[u])

                                        if all_pos_eff[u] in self.uncertain_positive_effects[a_name]:
                                            self.uncertain_positive_effects[a_name].remove(all_pos_eff[u])

                                        if all_pos_eff[u] in self.uncertain_negative_effects[a_name]:
                                            print("removing {} from possible negative effects of operator {}"
                                                  .format(all_pos_eff[u], a_name))
                                            self.uncertain_negative_effects[a_name].remove(all_pos_eff[u])

                            # Detect certain negative effects:
                            for u, eff in enumerate(all_neg_eff_obj):

                                # # If there are no inconsistencies with the positive effects, and the negative
                                # # effect appears only once in all negative effects, then it is a certain one
                                # if eff not in all_pos_eff_obj and sum([eff == eff2 for eff2 in all_neg_eff_obj]) == 1\
                                #         and negative_effects[u] not in self.certain_negative_effects[a_name]:
                                #     print("Operator {}, adding certain negative effect {}".format(a_name, negative_effects[u]))
                                #     self.certain_negative_effects[a_name].append(negative_effects[u])
                                #
                                #     if negative_effects[u].replace("(not","").strip()[:-1] in self.possible_negative_effects[a_name]:
                                #         self.possible_negative_effects[a_name].remove(negative_effects[u].replace("(not","").strip()[:-1])
                                #
                                #     if negative_effects[u].replace("(not","").strip()[:-1] in self.possible_positive_effects[a_name]:
                                #         self.possible_positive_effects[a_name].remove(negative_effects[u].replace("(not","").strip()[:-1])

                                # If there are no inconsistencies with the positive effects, and the negative
                                # effect appears only once in all negative effects, then it is a certain one
                                if eff not in all_pos_eff_obj and sum([eff == eff2 for eff2 in all_neg_eff_obj]) == 1\
                                        and all_neg_eff[u] not in self.certain_negative_effects[a_name]\
                                        and (eff in old_state and eff not in current_state_pddl):

                                    eff_params = eff.strip()[1:-1].split()[1:]
                                    correctness = True
                                    for obj in eff_params:
                                        if sum([obj == a_obj for a_obj in a_params]) > 1:
                                            correctness = False
                                    if correctness:
                                        print("Operator {}, adding certain negative effect {}".format(a_name, all_neg_eff[u]))
                                        self.certain_negative_effects[a_name].append(all_neg_eff[u])

                                        if all_neg_eff[u].replace("(not","").strip()[:-1] in self.uncertain_negative_effects[a_name]:
                                            self.uncertain_negative_effects[a_name].remove(all_neg_eff[u].replace("(not", "").strip()[:-1])

                                        if all_neg_eff[u].replace("(not","").strip()[:-1] in self.uncertain_positive_effects[a_name]:
                                            self.uncertain_positive_effects[a_name].remove(all_neg_eff[u].replace("(not", "").strip()[:-1])



                            break

            [f.write(el + "\n") for el in data]

            if a_name not in self.operator_learned:
                self.operator_learned.append(a_name)



        # Update domain dummy (i.e., domain with only learned operators)
        shutil.copyfile("PDDL/domain_learned.pddl", os.path.join("PDDL/domain_dummy.pddl"))
        with open("PDDL/domain_dummy.pddl", 'r') as f:
            data = [el for el in f.read().split("\n") if el.strip() != ""]

            removed_rows = []

            for m in range(len(data)):

                row = data[m]

                if row.lower().strip().startswith("(:action"):

                    if data[m + 2].strip().startswith(":precondition") and data[m + 3].strip() == ")":
                        removed_rows.append(m)

                        for n in range(m + 1, len(data) - 1):
                            if data[n].strip().startswith("(:action"):
                                break
                            removed_rows.append(n)

            dummy_domain = [data[i] for i in range(len(data)) if i not in removed_rows]

        with open("PDDL/domain_dummy.pddl", 'w') as f:
            [f.write(el + "\n") for el in dummy_domain]

        # return action_model_updated
        return effects_updated


    def goal_reached(self):

        with open("PDDL/facts.pddl", "r") as f:
            data = [el.strip() for el in f.read().split("\n")]
            facts = re.findall(":init.*\(:goal","".join(data))[0]

        current_state_pddl = sorted(re.findall("\([^()]*\)", facts))

        with open("PDDL/facts.pddl", "r") as f:
            data = [el.strip() for el in f.read().split("\n")]
            facts = re.findall(":goal.*","".join(data))[0]

        goal = sorted(re.findall("\([^()]*\)", facts))

        return set(goal).issubset(set(current_state_pddl))


    def update_executability_constr(self, op, op_preconds):

        # Remove constraint predicates which are not in the operator preconditions (superset)
        for i in range(len(self.operator_executability_constr[op])):
            constr_pred = self.operator_executability_constr[op][i]

            involved_constr_predicates = sorted(re.findall("\([^()]*\)", " ".join(constr_pred)))

            involved_op_precond_predicates = sorted(re.findall("\([^()]*\)", " ".join(op_preconds)))

            negligible_predicates = [el for el in involved_constr_predicates if el not in involved_op_precond_predicates]

            self.operator_executability_constr[op][i] = [pred for pred in constr_pred if
                                                         (pred and pred.replace("not(", "")[:-1]) not in negligible_predicates]


        # Add precondition superset negation executability constraint
        involved_op_precond_predicates = sorted(re.findall("\([^()]*\)", " ".join(op_preconds)))
        self.operator_executability_constr[op].append(["not({})".format(pred) for pred in involved_op_precond_predicates])

        # Remove redundant constraints
        sorted_op_executability_constr = sorted(self.operator_executability_constr[op], key=len)

        to_remove = []

        for i in range(len(sorted_op_executability_constr) - 1):

            constr = sorted_op_executability_constr[i]

            for j in range(i + 1, len(sorted_op_executability_constr)):

                if set(constr).issubset(sorted_op_executability_constr[j]):
                    to_remove.append(sorted_op_executability_constr[j])

        for redundant_constr in to_remove:

            if redundant_constr in self.operator_executability_constr[op]:
                self.operator_executability_constr[op].remove(redundant_constr)


        # Check single predicate constraints of an operator, if a constraint contains only one negated predicate,
        # then the predicate is a necessary precondition of the operator

        for constr in self.operator_executability_constr[op]:

            if len(constr) == 1 and constr[0].strip().lower().startswith("not("):

                precondition = constr[0].strip().replace("not(","")[:-1]

                if precondition not in self.operator_certain_predicates[op]:
                    self.operator_uncertain_predicates[op].remove(precondition)
                    self.operator_certain_predicates[op].append(precondition)

                    # DEBUG
                    print("operator {}, adding certain precondition: {}".format(op, precondition))



    def check_feasibile_preconditions_negation(self, op_name, negated_precond):

        # Compute subgoal by negating checked preconditions in conjunction with all other preconditions of
        # refined operator (e.g. subgoal = (p1 & p2 & p3 & !p4 & !p5))
        op_preconds = self.get_operator_preconditions(op_name)
        op_params = self.get_operator_parameters(op_name)

        subgoal = ""

        for checked_precs in negated_precond:
            positive_preconds = [el for el in op_preconds if el not in checked_precs]

            subgoal_preconds = positive_preconds + ["(not {})".format(prec) for prec in checked_precs]

            subgoal += "(exists {} (and {}))\n".format(op_params, " ".join(subgoal_preconds))

        # DEBUG
        print("\n\nChecking feasibility of operator {} negated preconditions of length {}".format(op_name, len(negated_precond[0])))


        shutil.copyfile("PDDL/facts.pddl", os.path.join("PDDL/facts_dummy.pddl"))

        with open("PDDL/facts_dummy.pddl", 'r') as f:
            data = f.read().split("\n")
            for i in range(len(data)):
                row = data[i]

                if row.strip().find("(:goal") != -1:
                    end_index = i + 1

                    if data[i].strip().startswith(")"):
                        data[i] = ")\n(:goal (or \n{}) \n))".format(subgoal)
                    else:
                        data[i] = "(:goal (or \n{}) \n))".format(subgoal)

        with open("PDDL/facts_dummy.pddl", 'w') as f:
            [f.write(el + "\n") for el in data[:end_index]]

        plan, found = Planner.FD_dummy()
        # plan, found = Planner.Madagascar("domain_dummy.pddl", "facts_dummy.pddl")

        feasibility = plan is not None

        # DEBUG
        print("The feasibility of operator {} negated preconditions of length {} is: {}".format(op_name,
                                                                                                len(negated_precond[0]),
                                                                                                feasibility))

        return plan, feasibility



    def get_object_types_hierarchy(self):
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



    def get_object_types_hierarchyOLD(self):
        with open("PDDL/domain_learned.pddl", 'r') as f:
            data = f.read().split("\n")

            objects_row = [el.replace(")","").strip()
                           for el in re.findall(":types.*\(:predicates","++".join(data))[0].replace(":types","").replace("(:predicates", "").split("++")
                           if el.strip() != ""]

            objects = defaultdict(list)

            start_index = 0
            for row in objects_row:
                current_index = len(objects['objects'])
                row = row.replace("(", "").replace(")", "")
                if row.find("- ") != -1:
                    [objects['objects'].append(el) for el in row.strip().split("- ")[0].split()]
                    # objects[row.strip().split("- ")[1].strip()] = [el.strip()
                    #                                               for el in row.strip().split("- ")[0].strip().split()]
                    objects[row.strip().split("- ")[1].strip()].extend([el.strip()
                                                                  for el in row.strip().split("- ")[0].strip().split()])
                    start_index = current_index + 1
                else:
                    [objects['objects'].append(el) for el in row.split()]

            for object_key, object_values in objects.items():
                if object_key != 'objects':

                    for val in object_values:

                        for key in objects.keys():
                            if val == key:
                                objects[object_key] = [el for el in objects[object_key] + objects[val] if el != val]
        return objects











    def compute_illegal_actions(self, operator, executability_constr):

        # Copy input domain to a temp one
        shutil.copyfile("PDDL/domain_learned.pddl", "PDDL/domain_tmp.pddl")

        with open("PDDL/domain_tmp.pddl", "r") as f:
            data = f.read().split("\n")

        # Remove not learned operators

        with open("PDDL/domain_tmp.pddl", "w") as f:

            for i in range(len(data)):

                if data[i].find(":action") != -1:
                    for j in range(i, len(data)):
                        data[j] = ""
                    break

            data = [el for el in data if el.strip() != ""]

            executability_constr = sorted(re.findall("\([^()]*\)", " ".join(executability_constr)))

            executability_constr = ["(not {})".format(el) for el in executability_constr]

            data.append("\n(:action {}\n:parameters {}\n:precondition (and {})\n:effect ()\n)\n\n)"
                        .format(operator,
                                self.get_operator_parameters(operator),
                                " ".join(executability_constr)))

            [f.write(data[i] + "\n") for i in range(len(data))]

        with open("PDDL/domain_tmp.pddl", "r") as f:
            data = f.read().split("\n")

        with open("PDDL/domain_tmp.pddl", "w") as f:

            for i in range(len(data)):

                if data[i].find(":predicates") != -1:

                    all_obj = self.get_all_object()

                    all_obj_fict_preds = ["(appear_{} ?obj - {})".format(k, k) for k in all_obj.keys()]

                    data[i] = data[i] + "\n" + "\n".join(all_obj_fict_preds)

                    data[i] = data[i] + "\n(true )"

                elif data[i].find(":action") != -1:
                    op_params = [el for i, el in enumerate(data[i + 1].replace(":parameters", "").strip()[1:-1].split())
                                 if el.startswith("?")]

                    # op_params_types = [el for i,el in enumerate(data[i+1].replace(":parameters", "").strip()[1:-1].split())
                    #                    if not el.startswith("?") and el.strip() != "-"]

                    single_obj_count = 0
                    op_params_types = []
                    row = [el for el in data[i + 1].replace(":parameters", "").strip()[1:-1].split() if
                           el.strip() != "-"]
                    for el in row:
                        if el.startswith("?"):
                            single_obj_count += 1
                        else:
                            [op_params_types.append(el) for _ in range(single_obj_count)]
                            single_obj_count = 0

                    op_effect = data[i + 5].replace(":effect", "")

                    if op_effect.find("(and") != -1:
                        op_effect = op_effect.replace("(and ", "")
                        op_effect = op_effect.strip()[:-1]

                    fictitious_eff = ""

                    for param in op_params:

                        if " ".join(data[i + 2:i + 6]).find(param + ")") == -1 and " ".join(data[i + 2:i + 6]).find(
                                param + " ") == -1:
                            n = op_params.index(param)
                            fictitious_eff += "(appear_{} ?param_{})".format(op_params_types[n], n + 1)

                    # fictitious_eff = " ".join(["(appear_{} ?param_{})".format(op_params_types[n], n+1) for n in range(len(op_params_types))])

                    data[i + 3] = ":effect (and {})".format(fictitious_eff + op_effect)

            # Add fictitious action
            for i in range(len(data)):
                if data[i].find("(:action") != -1:
                    data[i] = "(:action fict\n:parameters ()\n:precondition(and)\n:effect(true))" + "\n" + data[i]
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

                    for j in range(i, len(data)):
                        data[j] = ""

                    data[i] = "(:goal (and (true))))"

            [f.write(el + "\n") for el in data]

        bash_command = "Planners/FF/ff -o PDDL/domain_tmp.pddl -f PDDL/facts_tmp.pddl -i 114  >> outputff.txt"

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

                        for j in range(i + 1, len(data)):
                            if data[j].find("-----------operator") != -1 or data[j].find(
                                    "Cueing down from goal distance") != -1:
                                break

                            action_obj = [el.lower() for k, el in enumerate(data[j].replace(",", "").split()) if
                                          k % 3 == 0][1:]

                            if len(action_obj) > 0:
                                action_labels.append("{}({})".format(op_name, ",".join(action_obj)))

        # print("(Preprocessing) -- Reading ADL2STRIPS finished!")

        action_labels = [el for el in action_labels if el in self.action_labels]

        # Remove FF files
        os.remove("PDDL/domain_tmp.pddl")
        os.remove("PDDL/facts_tmp.pddl")
        os.remove("outputff.txt")


        print("Removing {} actions from action labels list".format(len(action_labels)))
        [self.action_labels.remove(a) for a in action_labels]
        print("Total actions: {}".format(len(self.action_labels)))

        # If some actions have been removed
        if len(action_labels) > 0:
            self.not_executable_actions_index = []


    def get_all_object(self):

        with open("PDDL/facts.pddl", "r") as f:
            data = [el.strip() for el in f.read().split("\n") if not el.strip().startswith(";")]
            obj_list = re.findall(":objects.*:init", "++".join(data))[0].replace(":objects", "").replace("(:init", "")

        obj_list = [el.replace(")","") for el in obj_list.split("++") if el.strip() != "" and el.strip() != ")"]

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















    def finalize_positive_effects_unknown(self, simulator):

        print("\n\n\nLooking for new positive effects of learned operators")

        new_effects_added = False


        for operator in self.operator_learned:

            all_possible_effects = self.get_op_relevant_predicates(operator)

            # uncertain_effects = [e for e in all_possible_effects if e not in self.uncertain_positive_effects[operator]]

            uncertain_effects = self.uncertain_positive_effects[operator]

            op_learned_effects = self.get_operator_effects(operator)

            op_neg_effect = re.findall("\(not[^)]*\)\)", op_learned_effects)
            op_pos_effect = [el for el in re.findall("\([^()]*\)", op_learned_effects)
                              if el not in [el.replace("(not","").strip()[:-1] for el in op_neg_effect]
                              and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

            uncertain_effects = [e for e in uncertain_effects if e not in op_pos_effect
                                 and e not in [el.replace("(not","").strip()[:-1] for el in op_neg_effect]
                                 and e not in self.get_operator_preconditions(operator)]

            if len(uncertain_effects) == 0:
                print("Operator {} does not admit new positive effects".format(operator))


            else:


                for checked_pos_eff in uncertain_effects:

                    op_params = self.get_operator_parameters(operator)
                    op_preconds = self.get_operator_preconditions(operator)

                    # Avoid ambiguity in checked effects (otherwise the positive effect cannot be learned with certainty)
                    checked_eff_params = " ".join(checked_pos_eff.split()[1:])[:-1].strip().split()
                    op_types = [el for el in self.get_operator_parameters(operator).strip()[1:-1].split()
                                if not el.startswith("-")]
                    types = defaultdict(list)
                    param_of_types = []
                    for k in range(len(op_types)):
                        if op_types[k].startswith("?"):
                            param_of_types.append(op_types[k])
                        else:
                            types[op_types[k]].extend(param_of_types)
                            param_of_types = []

                    checked_eff_types = []
                    for par in checked_eff_params:
                        for k,v in types.items():
                            if par in v:
                                checked_eff_types.append(k)
                                break

                    not_equal_params = [[] for _ in checked_eff_params]
                    not_equal_constr = []

                    for k, p_type in enumerate(checked_eff_types):
                        for el in types[p_type]:
                            if el != checked_eff_params[k]:
                                not_equal_params[k].append(el)
                                not_equal_constr.append("(not (= {} {}))".format(checked_eff_params[k], el))

                    not_equal_constr = list(set(not_equal_constr))

                    subgoal = ""

                    subgoal += "(exists {} (and {} (not {}) {}))\n".format(op_params,
                                                                           " ".join(op_preconds),
                                                                           checked_pos_eff,
                                                                           " ".join(not_equal_constr))

                    print(subgoal)


                    # DEBUG
                    print("\n\nChecking feasibility of operator {} with possible positive effect {}".format(operator,
                                                                                                            checked_pos_eff))

                    # shutil.copyfile("PDDL/facts.pddl", os.path.join("PDDL/facts_dummy.pddl"))
                    #
                    # with open("PDDL/facts_dummy.pddl", 'r') as f:
                    #     data = f.read().split("\n")
                    #     for i in range(len(data)):
                    #         row = data[i]
                    #
                    #         if row.strip().startswith("(:goal"):
                    #             end_index = i + 1
                    #             data[i] = "(:goal (or \n{}) \n))".format(subgoal)
                    #
                    # with open("PDDL/facts_dummy.pddl", 'w') as f:
                    #     [f.write(el + "\n") for el in data[:end_index]]

                    updated_effects = True

                    while updated_effects:

                        shutil.copyfile("PDDL/facts.pddl", os.path.join("PDDL/facts_dummy.pddl"))

                        with open("PDDL/facts_dummy.pddl", 'r') as f:
                            data = f.read().split("\n")
                            for i in range(len(data)):
                                row = data[i]

                                if row.strip().startswith("(:goal"):
                                    end_index = i + 1
                                    data[i] = "(:goal (or \n{}) \n))".format(subgoal)

                        with open("PDDL/facts_dummy.pddl", 'w') as f:
                            [f.write(el + "\n") for el in data[:end_index]]

                        plan, found = Planner.FD_dummy()
                        # plan, found = Planner.Madagascar("domain_dummy.pddl", "facts_dummy.pddl")

                        feasibility = plan is not None

                        updated_effects = False

                    # # Execute found plan
                    # if feasibility:

                        if not feasibility:
                            break

                        # Execute found plan
                        plan_failed = False

                        for action in plan:

                            old_state = simulator.get_state()
                            obs, done = simulator.execute(action.lower())

                            # DEBUG
                            if done:
                                print("Successfully executed action {}: {}".format(self.action_labels.index(action.lower()),
                                                                                   action.lower()))

                                # Update action model effects
                                updated_effects = self.add_operator_effects(action.lower(), old_state, simulator.get_state())

                                self.tried_actions = []
                                self.not_executable_actions = []
                                self.not_executable_actions_index = []

                                # Evaluate online metrics for log file
                                new_now = default_timer()
                                self.time_at_iter.append(new_now - self.now)
                                self.now = new_now
                                self.iter += 1
                                self.eval_log()

                                self.last_action = self.action_labels.index(action.lower())

                                self.parser.update_pddl_facts(simulator.get_state())

                                if updated_effects:
                                    break

                            else:

                                # print('Something went wrong, the found plan should be executable, (otherwise it means the model'
                                #       ' is not safe)')
                                # exit()

                                print("The computed plan has failed since there is an unknown negative "
                                      "effects in a plan action and such negative effect cannot be learned "
                                      "due to at least a redundant constants in action input that is involved "
                                      "by the negative effect")

                                plan_failed = True



                        if not updated_effects and not plan_failed:

                            executable_actions = self.compute_executable_actions()


                            # Choose operator action to be executed
                            op_action = None

                            for action in [a for a in executable_actions if a.startswith("{}(".format(operator))]:

                                action_obj = action.replace(operator,"").strip()[1:-1].split(",")

                                checked_pos_eff_tmp = copy.deepcopy(checked_pos_eff)

                                for i in range(len(action_obj)):
                                    checked_pos_eff_tmp = checked_pos_eff_tmp.replace(" ?param_{} ".format(i+1), " {} ".format(action_obj[i]))
                                    checked_pos_eff_tmp = checked_pos_eff_tmp.replace(" ?param_{})".format(i+1), " {})".format(action_obj[i]))

                                if checked_pos_eff_tmp not in simulator.get_state():
                                    op_action = action
                                    break

                            if op_action is None:

                                print("WARNING: The computed plan has been succesfully executed but there is an unknown negative "
                                      "effects in a plan action and such negative effect cannot be learned "
                                      "due to at least a redundant constants in action input that is involved "
                                      "by the negative effect")

                            # assert op_action is not None, "Something went wrong, check known positive effects test method"

                            else:

                                old_state = simulator.get_state()
                                obs, done = simulator.execute(op_action.lower())

                                # DEBUG
                                if done:
                                    print("Successfully executed action {}: {}".format(self.action_labels.index(op_action.lower()),
                                                                                       op_action.lower()))

                                    # Update action model effects
                                    updated_effects = self.add_operator_effects(op_action.lower(), old_state, simulator.get_state())

                                    self.tried_actions = []
                                    self.not_executable_actions = []
                                    self.not_executable_actions_index = []

                                    # Evaluate online metrics for log file
                                    new_now = default_timer()
                                    self.time_at_iter.append(new_now - self.now)
                                    self.now = new_now
                                    self.iter += 1
                                    self.eval_log()

                                    self.last_action = self.action_labels.index(op_action.lower())

                                    self.parser.update_pddl_facts(simulator.get_state())

                                    # Check if checked effect is a positive one
                                    action_obj = op_action.replace(operator,"").strip()[1:-1].split(",")

                                    checked_pos_eff_tmp = copy.deepcopy(checked_pos_eff)

                                    for i in range(len(action_obj)):
                                        checked_pos_eff_tmp = checked_pos_eff_tmp.replace(" ?param_{} ".format(i+1), " {} ".format(action_obj[i]))
                                        checked_pos_eff_tmp = checked_pos_eff_tmp.replace(" ?param_{})".format(i+1), " {})".format(action_obj[i]))

                                    # if checked_pos_eff_tmp in simulator.get_state():
                                    if updated_effects:
                                        new_effects_added = True
                                    elif checked_pos_eff_tmp not in simulator.get_state() and \
                                            checked_pos_eff in self.uncertain_positive_effects[operator]:
                                        self.uncertain_positive_effects[operator].remove(checked_pos_eff)

        return new_effects_added















    def finalize_positive_effects_known(self, simulator):

        print("\n\n\n Checking known positive effects of learned operators")


        for operator in self.operator_learned:

            op_learned_effects = self.get_operator_effects(operator)

            op_neg_effect = re.findall("\(not[^)]*\)\)", op_learned_effects)
            op_pos_effect = [el for el in re.findall("\([^()]*\)", op_learned_effects)
                              if el not in [el.replace("(not","").strip()[:-1] for el in op_neg_effect]
                              and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

            op_pos_effect = [el for el in op_pos_effect if el not in self.certain_positive_effects[operator]]

            if len(op_pos_effect) == 0:
                print("Operator {} has no known positive effects to check".format(operator))

            else:


                for checked_pos_eff in op_pos_effect:

                    op_params = self.get_operator_parameters(operator)
                    op_preconds = self.get_operator_preconditions(operator)

                    subgoal = ""

                    subgoal += "(exists {} (and {} (not {})))\n".format(op_params, " ".join(op_preconds), checked_pos_eff)

                    # DEBUG
                    print("\n\nChecking feasibility of operator {} with known positive effect {}".format(operator,
                                                                                                            checked_pos_eff))

                    shutil.copyfile("PDDL/facts.pddl", os.path.join("PDDL/facts_dummy.pddl"))

                    with open("PDDL/facts_dummy.pddl", 'r') as f:
                        data = f.read().split("\n")
                        for i in range(len(data)):
                            row = data[i]

                            if row.strip().startswith("(:goal"):
                                end_index = i + 1
                                data[i] = "(:goal (or \n{}) \n))".format(subgoal)

                    with open("PDDL/facts_dummy.pddl", 'w') as f:
                        [f.write(el + "\n") for el in data[:end_index]]


                    effects_updated = True

                    while effects_updated:

                        plan, found = Planner.FD_dummy()
                        # plan, found = Planner.Madagascar("domain_dummy.pddl", "facts_dummy.pddl")


                        feasibility = plan is not None

                        effects_updated = False

                        if not feasibility:
                            break

                        # Execute found plan
                        # if feasibility:

                        for action in plan:

                            old_state = simulator.get_state()
                            obs, done = simulator.execute(action.lower())

                            # DEBUG
                            if done:
                                print("Successfully executed action {}: {}".format(self.action_labels.index(action.lower()),
                                                                                   action.lower()))

                                # Update action model effects
                                effects_updated = self.add_operator_effects(action.lower(), old_state, simulator.get_state())

                                self.tried_actions = []
                                self.not_executable_actions = []
                                self.not_executable_actions_index = []

                                # Evaluate online metrics for log file
                                new_now = default_timer()
                                self.time_at_iter.append(new_now - self.now)
                                self.now = new_now
                                self.iter += 1
                                self.eval_log()

                                self.last_action = self.action_labels.index(action.lower())

                                self.parser.update_pddl_facts(simulator.get_state())


                                if effects_updated:
                                    break


                            else:

                                print('Something went wrong, the found plan should be executable, (otherwise it means the model'
                                      ' is not safe)')

                                exit()

                        if not effects_updated:

                            executable_actions = self.compute_executable_actions()

                            op_action = None

                            for action in [a for a in executable_actions if a.startswith("{}(".format(operator))]:

                                action_obj = action.replace(operator,"").strip()[1:-1].split(",")

                                checked_pos_eff_tmp = copy.deepcopy(checked_pos_eff)

                                for i in range(len(action_obj)):
                                    checked_pos_eff_tmp = checked_pos_eff_tmp.replace(" ?param_{} ".format(i+1), " {} ".format(action_obj[i]))
                                    checked_pos_eff_tmp = checked_pos_eff_tmp.replace(" ?param_{})".format(i+1), " {})".format(action_obj[i]))

                                if checked_pos_eff_tmp not in simulator.get_state():
                                    op_action = action
                                    break

                            assert op_action is not None, "Something went wrong, check known positive effects test method"

                            old_state = simulator.get_state()
                            obs, done = simulator.execute(op_action.lower())

                            # DEBUG
                            if done:
                                print("Successfully executed action {}: {}".format(self.action_labels.index(op_action.lower()),
                                                                                   op_action.lower()))

                                # Update action model effects
                                self.add_operator_effects(op_action.lower(), old_state, simulator.get_state())

                                self.tried_actions = []
                                self.not_executable_actions = []
                                self.not_executable_actions_index = []

                                # Evaluate online metrics for log file
                                new_now = default_timer()
                                self.time_at_iter.append(new_now - self.now)
                                self.now = new_now
                                self.iter += 1
                                self.eval_log()

                                self.last_action = self.action_labels.index(op_action.lower())

                                self.parser.update_pddl_facts(simulator.get_state())

                                # Check if checked effect is a positive one
                                if checked_pos_eff_tmp not in simulator.get_state()\
                                        and checked_pos_eff in self.uncertain_positive_effects[operator]:
                                    self.uncertain_positive_effects[operator].remove(checked_pos_eff)
                            else:

                                print('Something went wrong, action chosen while testing known operator effects '
                                      'should be surely executable.')
                                exit()

        return False

    def finalize_negative_effects_known(self, simulator):

        print("\n\n\n Checking known negative effects of learned operators")

        for operator in self.operator_learned:

            op_learned_effects = self.get_operator_effects(operator)

            all_op_neg_effect = re.findall("\(not[^)]*\)\)", op_learned_effects)
            all_op_neg_effect = [el.replace("(not", "").strip()[:-1] for el in all_op_neg_effect]

            op_neg_effect = re.findall("\(not[^)]*\)\)", op_learned_effects)
            op_pos_effect = [el for el in re.findall("\([^()]*\)", op_learned_effects)
                             if el not in [el.replace("(not", "").strip()[:-1] for el in op_neg_effect]
                             and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

            op_neg_effect = [el for el in op_neg_effect if el not in self.certain_negative_effects[operator]]

            op_neg_effect = [el.replace("(not", "").strip()[:-1] for el in op_neg_effect]

            if len(op_neg_effect) == 0:
                print("Operator {} has no known negative effects to check".format(operator))


            else:

                for checked_neg_eff in op_neg_effect:

                    op_params = self.get_operator_parameters(operator)
                    op_preconds = self.get_operator_preconditions(operator)
                    checked_eff_params = " ".join(checked_neg_eff.split()[1:])[:-1].strip().split()

                    same_effect_diff_params = [el for el in all_op_neg_effect
                                               if el.split()[0][1:] == checked_neg_eff.split()[0][1:]
                                               and el != checked_neg_eff]
                    same_effect_diff_params = [" ".join(el.split()[1:])[:-1].strip().split()
                                               for el in same_effect_diff_params]

                    not_equal_params = [[] for _ in checked_eff_params]
                    not_equal_constr = []
                    for el in same_effect_diff_params:
                        for i, param in enumerate(el):
                            if checked_eff_params[i] != param:
                                not_equal_params[i].append(param)
                                not_equal_constr.append("(not (= {} {}))".format(checked_eff_params[i], param))



                    subgoal = ""

                    subgoal += "(exists {} (and {} {} {}))\n".format(op_params, " ".join(op_preconds),
                                                                     checked_neg_eff,
                                                                     " ".join(not_equal_constr))

                    # DEBUG
                    print("\n\nChecking feasibility of operator {} with known negative effect {}".format(operator,
                                                                                                         checked_neg_eff))



                    effects_removed = False
                    feasibility = None

                    while feasibility is None:

                        # Parse facts_dummy.pddl
                        shutil.copyfile("PDDL/facts.pddl", os.path.join("PDDL/facts_dummy.pddl"))

                        with open("PDDL/facts_dummy.pddl", 'r') as f:
                            data = f.read().split("\n")
                            for i in range(len(data)):
                                row = data[i]

                                if row.strip().startswith("(:goal"):
                                    end_index = i + 1
                                    data[i] = "(:goal (or \n{}) \n))".format(subgoal)

                        with open("PDDL/facts_dummy.pddl", 'w') as f:
                            [f.write(el + "\n") for el in data[:end_index]]

                        # Solve problem
                        plan, found = Planner.FD_dummy()
                        # plan, found = Planner.Madagascar("domain_dummy.pddl", "facts_dummy.pddl")

                        feasibility = plan is not None

                        if not feasibility:
                            break

                        # Execute found plan
                        if feasibility:

                            for action in plan:

                                if feasibility == None:
                                    break

                                old_state = simulator.get_state()
                                obs, done = simulator.execute(action.lower())

                                # DEBUG
                                if done:
                                    print("Successfully executed action {}: {}".format(
                                        self.action_labels.index(action.lower()),
                                        action.lower()))

                                    # Update action model effects
                                    effects_removed = self.add_operator_effects(action.lower(), old_state, simulator.get_state())

                                    self.tried_actions = []
                                    self.not_executable_actions = []
                                    self.not_executable_actions_index = []

                                    # Evaluate online metrics for log file
                                    new_now = default_timer()
                                    self.time_at_iter.append(new_now - self.now)
                                    self.now = new_now
                                    self.iter += 1
                                    self.eval_log()

                                    self.last_action = self.action_labels.index(action.lower())

                                    self.parser.update_pddl_facts(simulator.get_state())

                                    if effects_removed:
                                        feasibility = None

                                else:

                                    print(
                                        'Something went wrong, the found plan should be executable, (otherwise it means the model'
                                        ' is not safe)')

                                    exit()

                    if feasibility:
                        executable_actions = self.compute_executable_actions()

                        op_action = None

                        for action in [a for a in executable_actions if a.startswith("{}(".format(operator))]:

                            action_obj = action.replace(operator, "").strip()[1:-1].split(",")

                            checked_neg_eff_tmp = copy.deepcopy(checked_neg_eff)

                            for i in range(len(action_obj)):
                                checked_neg_eff_tmp = checked_neg_eff_tmp.replace(" ?param_{} ".format(i + 1),
                                                                                  " {} ".format(action_obj[i]))
                                checked_neg_eff_tmp = checked_neg_eff_tmp.replace(" ?param_{})".format(i + 1),
                                                                                  " {})".format(action_obj[i]))

                            if checked_neg_eff_tmp in simulator.get_state():

                                # Check action objects are different according to constraint on parameters (i.e., some
                                # parameters must be different to avoid that the same ground effect corresponds to
                                # more than one lifted effect
                                change_action = False
                                for i, param in enumerate(checked_eff_params):
                                    for diff_param in not_equal_params[i]:
                                        if action_obj[i] == action_obj[int(diff_param.replace("?param_", "").strip())-1]:
                                            change_action = True

                                if not change_action:
                                    op_action = action
                                    break

                        assert op_action is not None, "Something went wrong, check known negative effects test method"

                        old_state = simulator.get_state()
                        obs, done = simulator.execute(op_action.lower())

                        # DEBUG
                        if done:
                            print("Successfully executed action {}: {}".format(
                                self.action_labels.index(op_action.lower()),
                                op_action.lower()))

                            # Update action model effects
                            self.add_operator_effects(op_action.lower(), old_state, simulator.get_state())

                            self.tried_actions = []
                            self.not_executable_actions = []
                            self.not_executable_actions_index = []

                            # Evaluate online metrics for log file
                            new_now = default_timer()
                            self.time_at_iter.append(new_now - self.now)
                            self.now = new_now
                            self.iter += 1
                            self.eval_log()

                            self.last_action = self.action_labels.index(op_action.lower())

                            self.parser.update_pddl_facts(simulator.get_state())

                            # Check if checked effect is a negative one
                            if checked_neg_eff_tmp in simulator.get_state() \
                                    and checked_neg_eff in self.uncertain_negative_effects[operator]:
                                print("removing {} from possible negative effects in finalizenegknown".format(checked_neg_eff))
                                self.uncertain_negative_effects[operator].remove(checked_neg_eff)
                        else:

                            print('Something went wrong, action chosen while testing known operator effects '
                                  'should be surely executable.')
                            exit()

        return False



    def finalize_negative_effects_unknown(self, simulator):

        print("\n\n\nLooking for new negative effects of learned operators")

        new_effects_added = False

        for operator in self.operator_learned:

            # all_possible_effects = self.get_op_relevant_predicates(operator)
            #
            # uncertain_effects_neg = [e for e in all_possible_effects if e not in self.uncertain_negative_effects[operator]]

            uncertain_effects_neg = self.uncertain_negative_effects[operator]

            op_learned_effects = self.get_operator_effects(operator)

            op_neg_effect = re.findall("\(not[^)]*\)\)", op_learned_effects)
            op_pos_effect = [el for el in re.findall("\([^()]*\)", op_learned_effects)
                              if el not in [el.replace("(not","").strip()[:-1] for el in op_neg_effect]
                              and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

            uncertain_effects_neg = [e for e in uncertain_effects_neg if e not in op_pos_effect
                                 and e not in [el.replace("(not","").strip()[:-1] for el in op_neg_effect]]


            # If a real negative effect is a positive precondition, then it is already always true whenever a
            # grounding of the operator is executable, hence it can be removed by the list of checked negative
            # effects
            # However this is not true if a negative effect is a precondition but cannot be learned because
            # it involve same action constants in a particular problem, hence it must be checked anyway
            # uncertain_effects_neg = [el for el in uncertain_effects_neg
            #                          if el not in self.get_operator_preconditions(operator)]


            if len(uncertain_effects_neg) == 0:
                    print("Operator {} does not admit new negative effects".format(operator))


            else:

                inconsistent_parameters_constraints = defaultdict(list)
                inconsistent_actions = []

                while len(uncertain_effects_neg) > 0:



                    subgoal = ""

                    op_params = self.get_operator_parameters(operator)
                    op_preconds = self.get_operator_preconditions(operator)

                    for checked_neg_eff in uncertain_effects_neg:

                        checked_neg_eff_pred = checked_neg_eff.strip()[1:-1].split()[0]
                        checked_neg_eff_params = checked_neg_eff.strip()[1:-1].split()[1:]

                        # if len(inconsistent_parameters_constraints[checked_neg_eff_pred]) > 0\
                        #         and any([el in inconsistent_parameters_constraints[checked_neg_eff_pred] for el in checked_neg_eff_params]):

                        # if len(inconsistent_parameters_constraints[checked_neg_eff_pred]) > 0:

                        if len(inconsistent_parameters_constraints[checked_neg_eff_pred]) == 1 :
                            # and any([el in inconsistent_parameters_constraints[checked_neg_eff_pred][0] for el in checked_neg_eff_params]):

                                all_eq_comb = list(itertools.combinations(inconsistent_parameters_constraints[checked_neg_eff_pred][0], 2))

                                # incons_equality_constr = "(not (= {}))".format(" ".join(inconsistent_parameters_constraints[checked_neg_eff_pred]))
                                incons_equality_constr = " ".join(["(not (= {}))".format(" ".join(el)) for el in all_eq_comb])

                                subgoal += "\n(exists {} (and {} {} {}))".format(op_params,
                                                                                 " ".join(op_preconds),
                                                                                 checked_neg_eff,
                                                                                 incons_equality_constr)

                        elif len(inconsistent_parameters_constraints[checked_neg_eff_pred]) > 1:
                            # and any([el in inconsistent_parameters_constraints[checked_neg_eff_pred][v]
                            #          for el in checked_neg_eff_params for v in range(len(inconsistent_parameters_constraints[checked_neg_eff_pred]))]):
                                incons_equality_constr = "(or "
                                for comb in inconsistent_parameters_constraints[checked_neg_eff_pred]:
                                    all_eq_comb = list(itertools.combinations(comb, 2))

                                    # incons_equality_constr = "(not (= {}))".format(" ".join(inconsistent_parameters_constraints[checked_neg_eff_pred]))
                                    incons_equality_constr += " (and {})".format(" ".join(["(not (= {}))".format(" ".join(el)) for el in all_eq_comb]))

                                incons_equality_constr += " )"
                                subgoal += "\n(exists {} (and {} {} {}))".format(op_params,
                                                                                 " ".join(op_preconds),
                                                                                 checked_neg_eff,
                                                                                 incons_equality_constr)


                        else:

                            subgoal += "\n(exists {} (and {} {}))".format(op_params, " ".join(op_preconds), checked_neg_eff)

                    # DEBUG
                    print("\n\nChecking feasibility of operator {} with possible negative effects".format(operator))

                    # shutil.copyfile("PDDL/facts.pddl", os.path.join("PDDL/facts_dummy.pddl"))
                    #
                    # with open("PDDL/facts_dummy.pddl", 'r') as f:
                    #     data = f.read().split("\n")
                    #     for i in range(len(data)):
                    #         row = data[i]
                    #
                    #         if row.strip().startswith("(:goal"):
                    #             end_index = i + 1
                    #             data[i] = "(:goal (or \n{}) \n))".format(subgoal)
                    #
                    # with open("PDDL/facts_dummy.pddl", 'w') as f:
                    #     [f.write(el + "\n") for el in data[:end_index]]


                    updated_effects = True

                    while updated_effects:

                        shutil.copyfile("PDDL/facts.pddl", os.path.join("PDDL/facts_dummy.pddl"))

                        with open("PDDL/facts_dummy.pddl", 'r') as f:
                            data = f.read().split("\n")
                            for i in range(len(data)):
                                row = data[i]

                                if row.strip().startswith("(:goal"):
                                    end_index = i + 1
                                    data[i] = "(:goal (or \n{}) \n))".format(subgoal)

                        with open("PDDL/facts_dummy.pddl", 'w') as f:
                            [f.write(el + "\n") for el in data[:end_index]]

                        plan, found = Planner.FD_dummy()
                        # plan, found = Planner.Madagascar("domain_dummy.pddl", "facts_dummy.pddl")

                        feasibility = plan is not None

                        updated_effects = False

                        if not feasibility:
                            break

                        # Execute found plan
                        # if feasibility:

                        for action in plan:

                            old_state = simulator.get_state()
                            obs, done = simulator.execute(action.lower())

                            # DEBUG
                            if done:
                                print("Successfully executed action {}: {}".format(self.action_labels.index(action.lower()),
                                                                                   action.lower()))

                                # Update action model effects
                                updated_effects = self.add_operator_effects(action.lower(), old_state, simulator.get_state())

                                self.tried_actions = []
                                self.not_executable_actions = []
                                self.not_executable_actions_index = []

                                # Evaluate online metrics for log file
                                new_now = default_timer()
                                self.time_at_iter.append(new_now - self.now)
                                self.now = new_now
                                self.iter += 1
                                self.eval_log()

                                self.last_action = self.action_labels.index(action.lower())

                                self.parser.update_pddl_facts(simulator.get_state())


                                if updated_effects:
                                    break


                            else:

                                print("Cannot execute {}".format(action.lower()))

                                print('Something went wrong, the found plan should be executable, (otherwise it means the model'
                                      ' is not safe)')

                                exit()


                        if not updated_effects:

                            executable_actions = self.compute_executable_actions()


                            # Choose operator action to be executed
                            op_action = None

                            all_checked_neg_effects = copy.deepcopy(uncertain_effects_neg)

                            all_actions_neg_eff_obj = []
                            all_actions_neg_eff_params = []

                            all_op_exec_actions = [a for a in executable_actions if a.startswith("{}(".format(operator))
                                                   if a not in inconsistent_actions]

                            for action in all_op_exec_actions:

                                action_obj = action.replace(operator, "").strip()[1:-1].split(",")

                                action_neg_eff_obj = []
                                action_neg_eff_params = []

                                for checked_neg_eff in all_checked_neg_effects:
                                    checked_neg_eff_tmp = copy.deepcopy(checked_neg_eff)

                                    for i in range(len(action_obj)):
                                        checked_neg_eff_tmp = checked_neg_eff_tmp.replace(" ?param_{} ".format(i + 1),
                                                                                          " {} ".format(action_obj[i]))
                                        checked_neg_eff_tmp = checked_neg_eff_tmp.replace(" ?param_{})".format(i + 1),
                                                                                          " {})".format(action_obj[i]))

                                    if checked_neg_eff_tmp in simulator.get_state():
                                        action_neg_eff_obj.append(checked_neg_eff_tmp)
                                        action_neg_eff_params.append(checked_neg_eff)

                                all_actions_neg_eff_obj.append(action_neg_eff_obj)
                                all_actions_neg_eff_params.append(action_neg_eff_params)

                            op_action = all_op_exec_actions[np.argmax([len(el) for el in all_actions_neg_eff_obj])]
                            op_action_neg_eff_obj = all_actions_neg_eff_obj[np.argmax([len(el) for el in all_actions_neg_eff_obj])]
                            op_action_neg_eff_params = all_actions_neg_eff_params[np.argmax([len(el) for el in all_actions_neg_eff_obj])]

                            # Execute chosen action
                            old_state = simulator.get_state()
                            obs, done = simulator.execute(op_action.lower())

                            # DEBUG
                            if done:
                                print("Successfully executed action {}: {}".format(self.action_labels.index(op_action.lower()),
                                                                                   op_action.lower()))

                                # Update action model effects
                                updated_effects = self.add_operator_effects(op_action.lower(), old_state, simulator.get_state())

                                self.tried_actions = []
                                self.not_executable_actions = []
                                self.not_executable_actions_index = []

                                # Evaluate online metrics for log file
                                new_now = default_timer()
                                self.time_at_iter.append(new_now - self.now)
                                self.now = new_now
                                self.iter += 1
                                self.eval_log()

                                self.last_action = self.action_labels.index(op_action.lower())

                                self.parser.update_pddl_facts(simulator.get_state())

                                # Check if there is ambiguity among checked effects, i.e., if the same object appears more
                                # than once in the same predicate (es. clean(obj1) and clean(obj1))
                                removed = []
                                inconsistent_preds = []
                                for i in range(len(op_action_neg_eff_obj)):
                                    for j in range(len(op_action_neg_eff_obj)):
                                        if op_action_neg_eff_obj[i] == op_action_neg_eff_obj[j] and i!=j:
                                            removed.append(i)
                                            removed.append(j)
                                            inconsistent_preds.append(op_action_neg_eff_obj[i].strip()[1:-1].split()[0])

                                removed = sorted(list(set(removed)))
                                inconsistent_preds = list(set(inconsistent_preds))
                                removed.reverse()

                                # Add inconsistency constraint over predicate parameters (e.g. to avoid clean(obj1) and clean(obj1
                                # when they are groundings of clean(?param_1) and clean(?param_2) ==> ?param_1 must not
                                # be equal to ?param_2

                                # for pred in inconsistent_preds:
                                #     inconsistent_params = []
                                #     for i in removed:
                                #
                                #         if op_action_neg_eff_params[i].strip()[1:-1].split()[0] == pred:
                                #             inconsistent_params.extend(op_action_neg_eff_params[i].strip()[1:-1].split()[1:])
                                #
                                #     inconsistent_parameters_constraints[pred] = list(set(inconsistent_params))

                                for pred in inconsistent_preds:

                                    all_pred_inconsistent = [op_action_neg_eff_params[i] for i in removed
                                                         if op_action_neg_eff_params[i].startswith("({} ".format(pred))]

                                    all_pred_inconsistent_params = [incons_pred.strip()[1:-1].split()[1:]
                                                                    for incons_pred in all_pred_inconsistent]
                                    not_eq_params = []
                                    for i in range(len(all_pred_inconsistent_params[0])):
                                        params = list(set([el[i] for el in all_pred_inconsistent_params]))

                                        if len(params) > 1 and params not in not_eq_params:
                                            not_eq_params.append(params)

                                    inconsistent_parameters_constraints[pred] = not_eq_params

                                    # inconsistent_params = []
                                    # for i in removed:
                                    #
                                    #     if op_action_neg_eff_params[i].strip()[1:-1].split()[0] == pred:
                                    #         inconsistent_params.extend(op_action_neg_eff_params[i].strip()[1:-1].split()[1:])
                                    #
                                    # inconsistent_parameters_constraints[pred] = list(set(inconsistent_params))


                                for i in removed:
                                    del op_action_neg_eff_obj[i]
                                    del op_action_neg_eff_params[i]

                                if len(removed) > 0:
                                    inconsistent_actions.append(op_action)


                                # Check if checked effect is a negative one
                                for k, checked_eff_obj in enumerate(op_action_neg_eff_obj):

                                    uncertain_effects_neg.remove(op_action_neg_eff_params[k])

                                    # if checked_eff_obj not in simulator.get_state():
                                    if updated_effects:
                                        new_effects_added = True
                                    elif checked_eff_obj in simulator.get_state() and \
                                            op_action_neg_eff_params[k] in self.uncertain_negative_effects[operator]:
                                        print("removing {} from possible negative effects in finalizenegunknown".format(op_action_neg_eff_params[k]))
                                        self.uncertain_negative_effects[operator].remove(op_action_neg_eff_params[k])

                            else:
                                print('Something went wrong, chosen action while testing effects should be executable')
                                exit()

                    # No more negative effects can be checked
                    if not feasibility:
                    # else:
                        break

        return new_effects_added


    def add_not_learn_op_precs(self):
        not_learn_ops = [o for o in self.all_operators if o not in self.operator_learned]

        for op in not_learn_ops:
            self.add_all_operator_precs(op)

