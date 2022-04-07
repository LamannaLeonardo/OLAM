# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import re
from collections import defaultdict

import Configuration




def action_model_statistics():

    real_precs_size, learned_precs_size = action_model_preconditions_size()
    real_eff_pos_size, learned_eff_pos_size = action_model_eff_pos_size()
    real_eff_neg_size, learned_eff_neg_size = action_model_eff_neg_size()

    ins_pre, del_pre = action_model_preconditions_statistics()

    ins_eff_pos, del_eff_pos = action_model_eff_pos_statistics()

    ins_eff_neg, del_eff_neg = action_model_eff_neg_statistics()

    precs_recall = action_model_prec_recall()

    eff_pos_recall = action_model_eff_pos_recall()

    eff_neg_recall = action_model_eff_neg_recall()

    precs_precision = action_model_prec_precision()

    eff_pos_precision = action_model_eff_pos_precision()

    eff_neg_precision = action_model_eff_neg_precision()

    overall_recall = action_model_overall_recall()

    overall_precision = action_model_overall_precision()

    return real_precs_size, learned_precs_size, real_eff_pos_size, learned_eff_pos_size, \
           real_eff_neg_size, learned_eff_neg_size, ins_pre, del_pre, ins_eff_pos, del_eff_pos, \
           ins_eff_neg, del_eff_neg, precs_recall, eff_pos_recall, eff_neg_recall, precs_precision, \
           eff_pos_precision, eff_neg_precision, overall_recall, overall_precision


def action_model_statistics_with_uncertain_neg(uncert_neg_effects):

    real_precs_size, learned_precs_size = action_model_preconditions_size()
    real_eff_pos_size, learned_eff_pos_size = action_model_eff_pos_size()
    real_eff_neg_size, learned_eff_neg_size = action_model_eff_neg_size()

    uncert_neg_eff_count = 0
    for k,v in uncert_neg_effects.items():
        uncert_neg_eff_count += len(v)

    learned_eff_neg_size += uncert_neg_eff_count

    ins_pre, del_pre = action_model_preconditions_statistics()

    ins_eff_pos, del_eff_pos = action_model_eff_pos_statistics()

    ins_eff_neg, del_eff_neg = action_model_eff_neg_statistics_with_uncertain(uncert_neg_effects)

    precs_recall = action_model_prec_recall()

    eff_pos_recall = action_model_eff_pos_recall()

    eff_neg_recall = action_model_eff_neg_recall_with_uncertain(uncert_neg_effects)

    precs_precision = action_model_prec_precision()

    eff_pos_precision = action_model_eff_pos_precision()

    eff_neg_precision = action_model_eff_neg_precision_with_uncertain(uncert_neg_effects)

    overall_recall = action_model_overall_recall_with_uncertain_neg(uncert_neg_effects)

    overall_precision = action_model_overall_precision_with_uncertain_neg(uncert_neg_effects)

    return real_precs_size, learned_precs_size, real_eff_pos_size, learned_eff_pos_size, \
           real_eff_neg_size, learned_eff_neg_size, ins_pre, del_pre, ins_eff_pos, del_eff_pos, \
           ins_eff_neg, del_eff_neg, precs_recall, eff_pos_recall, eff_neg_recall, precs_precision, \
           eff_pos_precision, eff_neg_precision, overall_recall, overall_precision

def action_model_prec_recall():

    tp_precs, fp_precs, fn_precs = action_model_preconditions_predictions()

    if (tp_precs + fn_precs) == 0:
        return 0

    return tp_precs / (tp_precs + fn_precs)


def action_model_prec_precision():

    tp_precs, fp_precs, fn_precs = action_model_preconditions_predictions()

    if (tp_precs + fp_precs) == 0:
        return 0

    return tp_precs / (tp_precs + fp_precs)



def action_model_eff_recall():

    tp_eff, fp_eff, fn_eff = action_model_eff_predictions()

    if (tp_eff + fn_eff) == 0:
        return 0

    return tp_eff / (tp_eff + fn_eff)



def action_model_eff_pos_recall():

    real_eff_pos_size, learned_eff_pos_size = action_model_eff_pos_size()

    if real_eff_pos_size == 0:
        return 1

    tp_eff_pos, fp_eff_pos, fn_eff_pos = action_model_eff_pos_predictions()

    if (tp_eff_pos + fn_eff_pos) == 0:
        return 0

    return tp_eff_pos / (tp_eff_pos + fn_eff_pos)



def action_model_eff_neg_recall():

    real_eff_neg_size, learned_eff_neg_size = action_model_eff_neg_size()

    if real_eff_neg_size == 0:
        return 1

    tp_eff_neg, fp_eff_neg, fn_eff_neg = action_model_eff_neg_predictions()

    if (tp_eff_neg + fn_eff_neg) == 0:
        return 0

    return tp_eff_neg / (tp_eff_neg + fn_eff_neg)



def action_model_eff_neg_recall_with_uncertain(uncert_neg_eff):

    real_eff_neg_size, learned_eff_neg_size = action_model_eff_neg_size()

    for k,v in uncert_neg_eff.items():
        learned_eff_neg_size += len(v)

    if real_eff_neg_size == 0:
        return 1

    tp_eff_neg, fp_eff_neg, fn_eff_neg = action_model_eff_neg_predictions_with_uncert(uncert_neg_eff)

    if (tp_eff_neg + fn_eff_neg) == 0:
        return 0

    return tp_eff_neg / (tp_eff_neg + fn_eff_neg)



def action_model_eff_pos_precision():

    real_eff_pos_size, learned_eff_pos_size = action_model_eff_pos_size()

    if real_eff_pos_size == 0:
        return 1

    tp_eff_pos, fp_eff_pos, fn_eff_pos = action_model_eff_pos_predictions()

    if (tp_eff_pos + fp_eff_pos) == 0:
        return 0

    return tp_eff_pos / (tp_eff_pos + fp_eff_pos)



def action_model_eff_neg_precision():

    real_eff_neg_size, learned_eff_neg_size = action_model_eff_neg_size()

    if real_eff_neg_size == 0:
        return 1

    tp_eff_neg, fp_eff_neg, fn_eff_neg = action_model_eff_neg_predictions()

    if (tp_eff_neg + fp_eff_neg) == 0:
        return 0

    return tp_eff_neg / (tp_eff_neg + fp_eff_neg)



def action_model_eff_neg_precision_with_uncertain(uncert_neg_eff):

    real_eff_neg_size, learned_eff_neg_size = action_model_eff_neg_size()

    for k,v in uncert_neg_eff.items():
        learned_eff_neg_size += len(v)

    if real_eff_neg_size == 0:
        return 1

    tp_eff_neg, fp_eff_neg, fn_eff_neg = action_model_eff_neg_predictions_with_uncert(uncert_neg_eff)

    if (tp_eff_neg + fp_eff_neg) == 0:
        return 0

    return tp_eff_neg / (tp_eff_neg + fp_eff_neg)



def action_model_eff_precision():

    tp_eff, fp_eff, fn_eff = action_model_eff_predictions()

    if (tp_eff + fp_eff) == 0:
        return 0

    return tp_eff / (tp_eff + fp_eff)



def action_model_overall_precision():

    tp_eff, fp_eff, fn_eff = action_model_eff_predictions()

    tp_precs, fp_precs, fn_precs = action_model_preconditions_predictions()

    all_tp = tp_precs + tp_eff
    all_fp = fp_eff + fp_precs
    all_fn = fn_eff + fn_precs

    if (all_tp + all_fp) == 0:
        return 0

    return all_tp / (all_tp + all_fp)



def action_model_overall_precision_with_uncertain_neg(uncert_neg_eff):

    tp_eff, fp_eff, fn_eff = action_model_eff_predictions_with_uncertain_neg(uncert_neg_eff)

    tp_precs, fp_precs, fn_precs = action_model_preconditions_predictions()

    all_tp = tp_precs + tp_eff
    all_fp = fp_eff + fp_precs
    all_fn = fn_eff + fn_precs

    if (all_tp + all_fp) == 0:
        return 0

    return all_tp / (all_tp + all_fp)



def action_model_overall_recall():

    tp_eff, fp_eff, fn_eff = action_model_eff_predictions()

    tp_precs, fp_precs, fn_precs = action_model_preconditions_predictions()

    all_tp = tp_precs + tp_eff
    all_fp = fp_eff + fp_precs
    all_fn = fn_eff + fn_precs

    if (all_tp + all_fn) == 0:
        return 0

    return all_tp / (all_tp + all_fn)



def action_model_overall_recall_with_uncertain_neg(uncertain_neg_eff):

    tp_eff, fp_eff, fn_eff = action_model_eff_predictions_with_uncertain_neg(uncertain_neg_eff)

    tp_precs, fp_precs, fn_precs = action_model_preconditions_predictions()

    all_tp = tp_precs + tp_eff
    all_fp = fp_eff + fp_precs
    all_fn = fn_eff + fn_precs

    if (all_tp + all_fn) == 0:
        return 0

    return all_tp / (all_tp + all_fn)



def action_model_eff_neg_predictions():

    real_action_eff_neg = defaultdict(list)

    learned_action_eff_neg = defaultdict(list)

    # Compute action model coverage and overfitting
    with open("PDDL/domain_learned.pddl", "r") as f:
        learned_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

        # Store learned action model effects

        all_action_schema = " ".join(learned_action_model)[" ".join(learned_action_model).index(":action "):]
        # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
        action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if el.strip() != ""]

        for schema in action_schema:
            op_name = schema.split()[1]
            all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

            cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)
            cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                              if el not in [el.replace("(not","").strip()[:-1] for el in cur_neg_effect]
                              and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

            learned_action_eff_neg[op_name] = cur_neg_effect




        with open("PDDL/domain.pddl", "r") as f:
            real_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

            # Store real action model effects

            all_action_schema = " ".join(real_action_model)[" ".join(real_action_model).index(":action "):]
            # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
            action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if
                             el.strip() != ""]

            for schema in action_schema:
                op_name = schema.split()[1]
                op_params = [el for el in re.findall("\([^()]*\)", re.findall(":parameters.*:precondition", schema)[0])[0].strip()[1:-1].split()
                             if el.startswith("?")]

                all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

                cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)
                cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                                  if el not in [el.replace("(not", "").strip()[:-1] for el in cur_neg_effect]
                                  and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

                for k in range(len(cur_neg_effect)):

                    for j,param in enumerate(op_params):
                            # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))

                # for k in range(len(cur_pos_effect)):
                #
                #     for j,param in enumerate(op_params):
                #             # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                #             cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                #             cur_pos_effect[k] = cur_pos_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                #             cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))




                real_action_eff_neg[op_name] = cur_neg_effect

    tp_eff_neg = 0
    fp_eff_neg = 0
    fn_eff_neg = 0

    for key, value in real_action_eff_neg.items():
        for pred in value:
            if pred not in learned_action_eff_neg[key]:
                fn_eff_neg += 1

    for key, value in learned_action_eff_neg.items():
        for pred in value:
            if pred in real_action_eff_neg[key]:
                tp_eff_neg += 1
            else:
                fp_eff_neg += 1

    return tp_eff_neg, fp_eff_neg, fn_eff_neg





def action_model_eff_neg_predictions_with_uncert(uncert_neg_eff):

    real_action_eff_neg = defaultdict(list)

    learned_action_eff_neg = defaultdict(list)

    # Compute action model coverage and overfitting
    with open("PDDL/domain_learned.pddl", "r") as f:
        learned_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

        # Store learned action model effects

        all_action_schema = " ".join(learned_action_model)[" ".join(learned_action_model).index(":action "):]
        # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
        action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if el.strip() != ""]

        for schema in action_schema:
            op_name = schema.split()[1]
            all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

            cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)
            cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                              if el not in [el.replace("(not","").strip()[:-1] for el in cur_neg_effect]
                              and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

            learned_action_eff_neg[op_name] = cur_neg_effect + ["(not {})".format(el) for el in uncert_neg_eff[op_name]]




        with open("PDDL/domain.pddl", "r") as f:
            real_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

            # Store real action model effects

            all_action_schema = " ".join(real_action_model)[" ".join(real_action_model).index(":action "):]
            # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
            action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if
                             el.strip() != ""]

            for schema in action_schema:
                op_name = schema.split()[1]
                op_params = [el for el in re.findall("\([^()]*\)", re.findall(":parameters.*:precondition", schema)[0])[0].strip()[1:-1].split()
                             if el.startswith("?")]

                all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

                cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)
                cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                                  if el not in [el.replace("(not", "").strip()[:-1] for el in cur_neg_effect]
                                  and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

                for k in range(len(cur_neg_effect)):

                    for j,param in enumerate(op_params):
                            # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))

                # for k in range(len(cur_pos_effect)):
                #
                #     for j,param in enumerate(op_params):
                #             # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                #             cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                #             cur_pos_effect[k] = cur_pos_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                #             cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))




                real_action_eff_neg[op_name] = cur_neg_effect

    tp_eff_neg = 0
    fp_eff_neg = 0
    fn_eff_neg = 0

    for key, value in real_action_eff_neg.items():
        for pred in value:
            if pred not in learned_action_eff_neg[key]:
                fn_eff_neg += 1

    for key, value in learned_action_eff_neg.items():
        for pred in value:
            if pred in real_action_eff_neg[key]:
                tp_eff_neg += 1
            else:
                fp_eff_neg += 1

    return tp_eff_neg, fp_eff_neg, fn_eff_neg

def action_model_eff_pos_predictions():

    real_action_eff_pos = defaultdict(list)

    learned_action_eff_pos = defaultdict(list)

    # Compute action model coverage and overfitting
    with open("PDDL/domain_learned.pddl", "r") as f:
        learned_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

        # Store learned action model effects

        all_action_schema = " ".join(learned_action_model)[" ".join(learned_action_model).index(":action "):]
        # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
        action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if el.strip() != ""]

        for schema in action_schema:
            op_name = schema.split()[1]
            all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

            cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)

            cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)]

            for neg in cur_neg_effect:
                if neg.replace("(not", "").strip()[:-1] in cur_pos_effect:
                    cur_pos_effect.remove(neg.replace("(not", "").strip()[:-1])
            # cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
            #                   if el not in [el.replace("(not","").strip()[:-1] for el in cur_neg_effect]
            #                   and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

            learned_action_eff_pos[op_name] = cur_pos_effect




        with open("PDDL/domain.pddl", "r") as f:
            real_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

            # Store real action model effects

            all_action_schema = " ".join(real_action_model)[" ".join(real_action_model).index(":action "):]
            # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
            action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if
                             el.strip() != ""]

            for schema in action_schema:
                op_name = schema.split()[1]
                op_params = [el for el in re.findall("\([^()]*\)", re.findall(":parameters.*:precondition", schema)[0])[0].strip()[1:-1].split()
                             if el.startswith("?")]

                all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

                cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)

                cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                                  if "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

                for neg in cur_neg_effect:
                    if neg.replace("(not", "").strip()[:-1] in cur_pos_effect:
                        cur_pos_effect.remove(neg.replace("(not", "").strip()[:-1])

                # cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                #                   if el not in [el.replace("(not", "").strip()[:-1] for el in cur_neg_effect]
                #                   and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]
                # cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                #                   if el not in [el.replace("(not", "").strip()[:-1] for el in cur_neg_effect]
                #                   and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

                # for k in range(len(cur_neg_effect)):
                #
                #     for j,param in enumerate(op_params):
                #             # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                #             cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                #             cur_neg_effect[k] = cur_neg_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                #             cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))

                for k in range(len(cur_pos_effect)):

                    for j,param in enumerate(op_params):
                            # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                            cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                            cur_pos_effect[k] = cur_pos_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                            cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))




                real_action_eff_pos[op_name] = cur_pos_effect

    tp_eff_pos = 0
    fp_eff_pos = 0
    fn_eff_pos = 0

    for key, value in real_action_eff_pos.items():
        for pred in value:
            if pred not in learned_action_eff_pos[key]:
                fn_eff_pos += 1

    for key, value in learned_action_eff_pos.items():
        for pred in value:
            if pred in real_action_eff_pos[key]:
                tp_eff_pos += 1
            else:
                fp_eff_pos += 1

    return tp_eff_pos, fp_eff_pos, fn_eff_pos



def action_model_eff_predictions():

    real_action_eff = defaultdict(list)

    learned_action_eff = defaultdict(list)

    # Compute action model coverage and overfitting
    with open("PDDL/domain_learned.pddl", "r") as f:
        learned_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

        # Store learned action model effects

        all_action_schema = " ".join(learned_action_model)[" ".join(learned_action_model).index(":action "):]
        # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
        action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if el.strip() != ""]

        for schema in action_schema:
            op_name = schema.split()[1]
            all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

            cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)

            cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                              if "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

            for neg in cur_neg_effect:
                if neg.replace("(not", "").strip()[:-1] in cur_pos_effect:
                    cur_pos_effect.remove(neg.replace("(not", "").strip()[:-1])
            # cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
            #                   if el not in [el.replace("(not","").strip()[:-1] for el in cur_neg_effect]
            #                   and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

            learned_action_eff[op_name] = cur_neg_effect + cur_pos_effect




        with open("PDDL/domain.pddl", "r") as f:
            real_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

            # Store real action model effects

            all_action_schema = " ".join(real_action_model)[" ".join(real_action_model).index(":action "):]
            # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
            action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if
                             el.strip() != ""]

            for schema in action_schema:
                op_name = schema.split()[1]
                op_params = [el for el in re.findall("\([^()]*\)", re.findall(":parameters.*:precondition", schema)[0])[0].strip()[1:-1].split()
                             if el.startswith("?")]

                all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

                cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)


                cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                                  if "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

                for neg in cur_neg_effect:
                    if neg.replace("(not", "").strip()[:-1] in cur_pos_effect:
                        cur_pos_effect.remove(neg.replace("(not", "").strip()[:-1])
                # cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                #                   if el not in [el.replace("(not", "").strip()[:-1] for el in cur_neg_effect]
                #                   and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

                for k in range(len(cur_neg_effect)):

                    for j,param in enumerate(op_params):
                            # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))

                for k in range(len(cur_pos_effect)):

                    for j,param in enumerate(op_params):
                            # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                            cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                            cur_pos_effect[k] = cur_pos_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                            cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))




                real_action_eff[op_name] = cur_neg_effect + cur_pos_effect

    tp_eff = 0
    fp_eff = 0
    fn_eff = 0

    for key, value in real_action_eff.items():
        for pred in value:
            if pred not in learned_action_eff[key]:
                fn_eff += 1

    for key, value in learned_action_eff.items():
        for pred in value:
            if pred in real_action_eff[key]:
                tp_eff += 1
            else:
                fp_eff += 1

    return tp_eff, fp_eff, fn_eff



def action_model_eff_predictions_with_uncertain_neg(uncert_neg_eff):

    real_action_eff = defaultdict(list)

    learned_action_eff = defaultdict(list)

    # Compute action model coverage and overfitting
    with open("PDDL/domain_learned.pddl", "r") as f:
        learned_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

        # Store learned action model effects

        all_action_schema = " ".join(learned_action_model)[" ".join(learned_action_model).index(":action "):]
        # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
        action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if el.strip() != ""]

        for schema in action_schema:
            op_name = schema.split()[1]
            all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

            cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)

            cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                              if "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

            for neg in cur_neg_effect:
                if neg.replace("(not", "").strip()[:-1] in cur_pos_effect:
                    cur_pos_effect.remove(neg.replace("(not", "").strip()[:-1])
            # cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
            #                   if el not in [el.replace("(not","").strip()[:-1] for el in cur_neg_effect]
            #                   and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

            learned_action_eff[op_name] = cur_neg_effect + ["(not {})".format(el) for el in uncert_neg_eff[op_name]] + cur_pos_effect




        with open("PDDL/domain.pddl", "r") as f:
            real_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

            # Store real action model effects

            all_action_schema = " ".join(real_action_model)[" ".join(real_action_model).index(":action "):]
            # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
            action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if
                             el.strip() != ""]

            for schema in action_schema:
                op_name = schema.split()[1]
                op_params = [el for el in re.findall("\([^()]*\)", re.findall(":parameters.*:precondition", schema)[0])[0].strip()[1:-1].split()
                             if el.startswith("?")]

                all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

                cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)


                cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                                  if "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

                for neg in cur_neg_effect:
                    if neg.replace("(not", "").strip()[:-1] in cur_pos_effect:
                        cur_pos_effect.remove(neg.replace("(not", "").strip()[:-1])
                # cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                #                   if el not in [el.replace("(not", "").strip()[:-1] for el in cur_neg_effect]
                #                   and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

                for k in range(len(cur_neg_effect)):

                    for j,param in enumerate(op_params):
                            # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))

                for k in range(len(cur_pos_effect)):

                    for j,param in enumerate(op_params):
                            # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                            cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                            cur_pos_effect[k] = cur_pos_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                            cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))




                real_action_eff[op_name] = cur_neg_effect + cur_pos_effect

    tp_eff = 0
    fp_eff = 0
    fn_eff = 0

    for key, value in real_action_eff.items():
        for pred in value:
            if pred not in learned_action_eff[key]:
                fn_eff += 1

    for key, value in learned_action_eff.items():
        for pred in value:
            if pred in real_action_eff[key]:
                tp_eff += 1
            else:
                fp_eff += 1

    return tp_eff, fp_eff, fn_eff



def action_model_preconditions_predictions():

    # Compute action model coverage and overfitting
    with open("PDDL/domain_learned.pddl", "r") as f:
        learned_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

        benchmark_dir = dir

        real_action_precond = defaultdict(list)

        learned_action_precond = defaultdict(list)

        # Store learned action model preconditions

        for i in range(len(learned_action_model) - 2):

            line = learned_action_model[i]

            if line.strip().find("(:action ") != -1:

                found_precond = False
                action_name = line.strip().split()[1]
                action_precond = []

                for j in range(i + 1, len(learned_action_model) - 1):

                    if found_precond:
                        break

                    if learned_action_model[j].strip().find(":precondition") != -1:
                        found_precond = True
                        action_precond.append(learned_action_model[j])

                        for k in range(j + 1, len(learned_action_model)):

                            if learned_action_model[k].strip().find(":effect") != -1:
                                break

                            action_precond.append(learned_action_model[k].strip())

                learned_action_precond[action_name] = list(set([el.replace(" ", "")
                                                                for el in sorted(re.findall("\([^()]*\)", "".join(action_precond)))
                                                                if "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]))

        domain = Configuration.INSTANCE_DATA_PATH_PDDL.split("/")[-3]
        benchmark_dir = Configuration.INSTANCE_DATA_PATH_PDDL.split("/")[-2]

        with open("PDDL/domain.pddl") as r:
            real_action_model = [el.lower() for el in r.read().split('\n') if el.strip() != ""]

            for i in range(len(real_action_model) - 2):

                line = real_action_model[i]

                if line.strip().find("(:action ") != -1:

                    found_precond = False
                    action_name = line.strip().split()[1]
                    # action_params = [el.replace(" -", "").strip() for el in
                    #                  re.findall("\?[^ - ]* -", real_action_model[i + 1])]

                    action_params = [el for el in real_action_model[i + 1].replace("(","").replace(")","").strip().split()[1:]
                                     if el.startswith("?")]

                    action_precond = []

                    for j in range(i + 1, len(real_action_model) - 1):

                        if found_precond:
                            break

                        if real_action_model[j].strip().find(":precondition") != -1:
                            found_precond = True
                            action_precond.append(real_action_model[j])

                            for k in range(j + 1, len(real_action_model)):

                                if real_action_model[k].strip().find(":effect") != -1:
                                    break

                                action_precond.append(real_action_model[k])

                    # Replace action precondition objects name with "param_#"

                    for p in range(len(action_precond)):
                        for el in action_params:
                            # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                            action_precond[p] = action_precond[p].replace(" " + el + " ", " ?param_{} ".format(
                                action_params.index(el) + 1))
                            action_precond[p] = action_precond[p].replace("(" + el + " ", "(?param_{} ".format(
                                action_params.index(el) + 1))
                            action_precond[p] = action_precond[p].replace(" " + el + ")", " ?param_{})".format(
                                action_params.index(el) + 1))

                    real_action_precond[action_name] = list(set([el.replace(" ", "")
                                                                 for el in sorted(
                            re.findall("\([^()]*\)", "".join(action_precond)))]))

    tp_precs = 0
    fp_precs = 0
    fn_precs = 0

    for key, value in real_action_precond.items():
        for pred in value:
            if pred not in learned_action_precond[key]:
                fn_precs += 1

    for key, value in learned_action_precond.items():
        for pred in value:
            if pred in real_action_precond[key]:
                tp_precs += 1
            else:
                fp_precs += 1

    return tp_precs, fp_precs, fn_precs



def action_model_preconditions_statistics():

    # Compute action model coverage and overfitting
    with open("PDDL/domain_learned.pddl", "r") as f:
        learned_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

        benchmark_dir = dir

        real_action_precond = defaultdict(list)

        learned_action_precond = defaultdict(list)

        # Store learned action model preconditions

        for i in range(len(learned_action_model) - 2):

            line = learned_action_model[i]

            if line.strip().find("(:action ") != -1:

                found_precond = False
                action_name = line.strip().split()[1]
                action_precond = []

                for j in range(i + 1, len(learned_action_model) - 1):

                    if found_precond:
                        break

                    if learned_action_model[j].strip().find(":precondition") != -1:
                        found_precond = True
                        action_precond.append(learned_action_model[j])

                        for k in range(j + 1, len(learned_action_model)):

                            if learned_action_model[k].strip().find(":effect") != -1:
                                break

                            action_precond.append(learned_action_model[k].strip())

                learned_action_precond[action_name] = list(set([el.replace(" ", "")
                                                                for el in sorted(re.findall("\([^()]*\)", "".join(action_precond)))
                                                                if "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]))

        domain = Configuration.INSTANCE_DATA_PATH_PDDL.split("/")[-3]
        benchmark_dir = Configuration.INSTANCE_DATA_PATH_PDDL.split("/")[-2]

        with open("PDDL/domain.pddl") as r:
            real_action_model = [el.lower() for el in r.read().split('\n') if el.strip() != ""]

            for i in range(len(real_action_model) - 2):

                line = real_action_model[i]

                if line.strip().find("(:action ") != -1:

                    found_precond = False
                    action_name = line.strip().split()[1]
                    # action_params = [el.replace(" -", "").strip() for el in
                    #                  re.findall("\?[^ - ]* -", real_action_model[i + 1])]

                    action_params = [el for el in real_action_model[i + 1].replace("(","").replace(")","").strip().split()[1:]
                                     if el.startswith("?")]

                    action_precond = []

                    for j in range(i + 1, len(real_action_model) - 1):

                        if found_precond:
                            break

                        if real_action_model[j].strip().find(":precondition") != -1:
                            found_precond = True
                            action_precond.append(real_action_model[j])

                            for k in range(j + 1, len(real_action_model)):

                                if real_action_model[k].strip().find(":effect") != -1:
                                    break

                                action_precond.append(real_action_model[k])

                    # Replace action precondition objects name with "param_#"

                    for p in range(len(action_precond)):
                        for el in action_params:
                            # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                            action_precond[p] = action_precond[p].replace(" " + el + " ", " ?param_{} ".format(
                                action_params.index(el) + 1))
                            action_precond[p] = action_precond[p].replace("(" + el + " ", "(?param_{} ".format(
                                action_params.index(el) + 1))
                            action_precond[p] = action_precond[p].replace(" " + el + ")", " ?param_{})".format(
                                action_params.index(el) + 1))

                    real_action_precond[action_name] = list(set([el.replace(" ", "")
                                                                 for el in sorted(
                            re.findall("\([^()]*\)", "".join(action_precond)))]))

    # tp_precs = 0
    # fp_precs = 0
    # fn_precs = 0
    ins_pre = 0
    del_pre = 0

    for key, value in real_action_precond.items():
        for pred in value:
            if pred not in learned_action_precond[key]:
                ins_pre += 1

    for key, value in learned_action_precond.items():
        for pred in value:
            if pred not in real_action_precond[key]:
                del_pre += 1

    return ins_pre, del_pre





def action_model_preconditions_size():

    # Compute action model coverage and overfitting
    with open("PDDL/domain_learned.pddl", "r") as f:
        learned_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

        benchmark_dir = dir

        real_action_precond = defaultdict(list)

        learned_action_precond = defaultdict(list)

        # Store learned action model preconditions

        for i in range(len(learned_action_model) - 2):

            line = learned_action_model[i]

            if line.strip().find("(:action ") != -1:

                found_precond = False
                action_name = line.strip().split()[1]
                action_precond = []

                for j in range(i + 1, len(learned_action_model) - 1):

                    if found_precond:
                        break

                    if learned_action_model[j].strip().find(":precondition") != -1:
                        found_precond = True
                        action_precond.append(learned_action_model[j])

                        for k in range(j + 1, len(learned_action_model)):

                            if learned_action_model[k].strip().find(":effect") != -1:
                                break

                            action_precond.append(learned_action_model[k].strip())

                learned_action_precond[action_name] = list(set([el.replace(" ", "")
                                                                for el in sorted(re.findall("\([^()]*\)", "".join(action_precond)))
                                                                if "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]))

        domain = Configuration.INSTANCE_DATA_PATH_PDDL.split("/")[-3]
        benchmark_dir = Configuration.INSTANCE_DATA_PATH_PDDL.split("/")[-2]

        with open("PDDL/domain.pddl") as r:
            real_action_model = [el.lower() for el in r.read().split('\n') if el.strip() != ""]

            for i in range(len(real_action_model) - 2):

                line = real_action_model[i]

                if line.strip().find("(:action ") != -1:

                    found_precond = False
                    action_name = line.strip().split()[1]
                    # action_params = [el.replace(" -", "").strip() for el in
                    #                  re.findall("\?[^ - ]* -", real_action_model[i + 1])]

                    action_params = [el for el in real_action_model[i + 1].replace("(","").replace(")","").strip().split()[1:]
                                     if el.startswith("?")]

                    action_precond = []

                    for j in range(i + 1, len(real_action_model) - 1):

                        if found_precond:
                            break

                        if real_action_model[j].strip().find(":precondition") != -1:
                            found_precond = True
                            action_precond.append(real_action_model[j])

                            for k in range(j + 1, len(real_action_model)):

                                if real_action_model[k].strip().find(":effect") != -1:
                                    break

                                action_precond.append(real_action_model[k])

                    # Replace action precondition objects name with "param_#"

                    for p in range(len(action_precond)):
                        for el in action_params:
                            # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                            action_precond[p] = action_precond[p].replace(" " + el + " ", " ?param_{} ".format(
                                action_params.index(el) + 1))
                            action_precond[p] = action_precond[p].replace("(" + el + " ", "(?param_{} ".format(
                                action_params.index(el) + 1))
                            action_precond[p] = action_precond[p].replace(" " + el + ")", " ?param_{})".format(
                                action_params.index(el) + 1))

                    real_action_precond[action_name] = list(set([el.replace(" ", "")
                                                                 for el in sorted(
                            re.findall("\([^()]*\)", "".join(action_precond)))]))

    # tp_precs = 0
    # fp_precs = 0
    # fn_precs = 0
    real_precs_size = 0
    learned_precs_size = 0

    for key, value in real_action_precond.items():
        for pred in value:
            real_precs_size += 1

    for key, value in learned_action_precond.items():
        for pred in value:
            learned_precs_size += 1

    return real_precs_size, learned_precs_size




def action_model_eff_pos_size():

    real_action_eff_pos = defaultdict(list)

    learned_action_eff_pos = defaultdict(list)

    # Compute action model coverage and overfitting
    with open("PDDL/domain_learned.pddl", "r") as f:
        learned_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

        # Store learned action model effects

        all_action_schema = " ".join(learned_action_model)[" ".join(learned_action_model).index(":action "):]
        # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
        action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if el.strip() != ""]

        for schema in action_schema:
            op_name = schema.split()[1]
            all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

            cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)

            cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                              if "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

            for neg in cur_neg_effect:
                if neg.replace("(not", "").strip()[:-1] in cur_pos_effect:
                    cur_pos_effect.remove(neg.replace("(not", "").strip()[:-1])
            # cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
            #                   if el not in [el.replace("(not","").strip()[:-1] for el in cur_neg_effect]
            #                   and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

            learned_action_eff_pos[op_name] = cur_pos_effect




        with open("PDDL/domain.pddl", "r") as f:
            real_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

            # Store real action model effects

            all_action_schema = " ".join(real_action_model)[" ".join(real_action_model).index(":action "):]
            # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
            action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if
                             el.strip() != ""]

            for schema in action_schema:
                op_name = schema.split()[1]
                op_params = [el for el in re.findall("\([^()]*\)", re.findall(":parameters.*:precondition", schema)[0])[0].strip()[1:-1].split()
                             if el.startswith("?")]

                all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

                cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)


                cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                                  if "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

                for neg in cur_neg_effect:
                    if neg.replace("(not", "").strip()[:-1] in cur_pos_effect:
                        cur_pos_effect.remove(neg.replace("(not", "").strip()[:-1])
                # cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                #                   if el not in [el.replace("(not", "").strip()[:-1] for el in cur_neg_effect]
                #                   and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

                # for k in range(len(cur_neg_effect)):
                #
                #     for j,param in enumerate(op_params):
                #             # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                #             cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                #             cur_neg_effect[k] = cur_neg_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                #             cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))

                for k in range(len(cur_pos_effect)):

                    for j,param in enumerate(op_params):
                            # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                            cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                            cur_pos_effect[k] = cur_pos_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                            cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))




                real_action_eff_pos[op_name] = cur_pos_effect

    # tp_eff = 0
    # fp_eff = 0
    # fn_eff = 0
    real_size_eff_pos = 0
    learned_size_eff_pos = 0

    for key, value in real_action_eff_pos.items():
        for pred in value:
            real_size_eff_pos += 1

    for key, value in learned_action_eff_pos.items():
        for pred in value:
            learned_size_eff_pos += 1

    return real_size_eff_pos, learned_size_eff_pos



def action_model_eff_neg_size():

    real_action_eff_neg = defaultdict(list)

    learned_action_eff_neg = defaultdict(list)

    # Compute action model coverage and overfitting
    with open("PDDL/domain_learned.pddl", "r") as f:
        learned_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

        # Store learned action model effects

        all_action_schema = " ".join(learned_action_model)[" ".join(learned_action_model).index(":action "):]
        # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
        action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if el.strip() != ""]

        for schema in action_schema:
            op_name = schema.split()[1]
            all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

            cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)
            # cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
            #                   if el not in [el.replace("(not","").strip()[:-1] for el in cur_neg_effect]
            #                   and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

            learned_action_eff_neg[op_name] = cur_neg_effect




        with open("PDDL/domain.pddl", "r") as f:
            real_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

            # Store real action model effects

            all_action_schema = " ".join(real_action_model)[" ".join(real_action_model).index(":action "):]
            # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
            action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if
                             el.strip() != ""]

            for schema in action_schema:
                op_name = schema.split()[1]
                op_params = [el for el in re.findall("\([^()]*\)", re.findall(":parameters.*:precondition", schema)[0])[0].strip()[1:-1].split()
                             if el.startswith("?")]

                all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

                cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)
                # cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                #                   if el not in [el.replace("(not", "").strip()[:-1] for el in cur_neg_effect]
                #                   and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

                for k in range(len(cur_neg_effect)):

                    for j,param in enumerate(op_params):
                            # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))

                # for k in range(len(cur_pos_effect)):
                #
                #     for j,param in enumerate(op_params):
                #             # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                #             cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                #             cur_pos_effect[k] = cur_pos_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                #             cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))




                real_action_eff_neg[op_name] = cur_neg_effect

    # tp_eff = 0
    # fp_eff = 0
    # fn_eff = 0
    real_size_eff_neg = 0
    learned_size_eff_neg = 0

    for key, value in real_action_eff_neg.items():
        for pred in value:
            real_size_eff_neg += 1

    for key, value in learned_action_eff_neg.items():
        for pred in value:
            learned_size_eff_neg += 1

    return real_size_eff_neg, learned_size_eff_neg



def action_model_eff_pos_statistics():

    real_action_eff_pos = defaultdict(list)

    learned_action_eff_pos = defaultdict(list)

    # Compute action model coverage and overfitting
    with open("PDDL/domain_learned.pddl", "r") as f:
        learned_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

        # Store learned action model effects

        all_action_schema = " ".join(learned_action_model)[" ".join(learned_action_model).index(":action "):]
        # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
        action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if el.strip() != ""]

        for schema in action_schema:
            op_name = schema.split()[1]
            all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

            cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)

            cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                              if "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

            for neg in cur_neg_effect:
                if neg.replace("(not", "").strip()[:-1] in cur_pos_effect:
                    cur_pos_effect.remove(neg.replace("(not", "").strip()[:-1])
            # cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
            #                   if el not in [el.replace("(not","").strip()[:-1] for el in cur_neg_effect]
            #                   and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

            learned_action_eff_pos[op_name] = cur_pos_effect




        with open("PDDL/domain.pddl", "r") as f:
            real_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

            # Store real action model effects

            all_action_schema = " ".join(real_action_model)[" ".join(real_action_model).index(":action "):]
            # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
            action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if
                             el.strip() != ""]

            for schema in action_schema:
                op_name = schema.split()[1]
                op_params = [el for el in re.findall("\([^()]*\)", re.findall(":parameters.*:precondition", schema)[0])[0].strip()[1:-1].split()
                             if el.startswith("?")]

                all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

                cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)


                cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                                  if "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

                for neg in cur_neg_effect:
                    if neg.replace("(not", "").strip()[:-1] in cur_pos_effect:
                        cur_pos_effect.remove(neg.replace("(not", "").strip()[:-1])
                # cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                #                   if el not in [el.replace("(not", "").strip()[:-1] for el in cur_neg_effect]
                #                   and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

                # for k in range(len(cur_neg_effect)):
                #
                #     for j,param in enumerate(op_params):
                #             # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                #             cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                #             cur_neg_effect[k] = cur_neg_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                #             cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))

                for k in range(len(cur_pos_effect)):

                    for j,param in enumerate(op_params):
                            # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                            cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                            cur_pos_effect[k] = cur_pos_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                            cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))




                real_action_eff_pos[op_name] = cur_pos_effect

    # tp_eff = 0
    # fp_eff = 0
    # fn_eff = 0
    ins_add = 0
    del_add = 0

    for key, value in real_action_eff_pos.items():
        for pred in value:
            if pred not in learned_action_eff_pos[key]:
                ins_add += 1

    for key, value in learned_action_eff_pos.items():
        for pred in value:
            if pred not in real_action_eff_pos[key]:
                del_add += 1

    return ins_add, del_add



def action_model_eff_neg_statistics():

    real_action_eff_neg = defaultdict(list)

    learned_action_eff_neg = defaultdict(list)

    # Compute action model coverage and overfitting
    with open("PDDL/domain_learned.pddl", "r") as f:
        learned_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

        # Store learned action model effects

        all_action_schema = " ".join(learned_action_model)[" ".join(learned_action_model).index(":action "):]
        # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
        action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if el.strip() != ""]

        for schema in action_schema:
            op_name = schema.split()[1]
            all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

            cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)
            # cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
            #                   if el not in [el.replace("(not","").strip()[:-1] for el in cur_neg_effect]
            #                   and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

            learned_action_eff_neg[op_name] = cur_neg_effect




        with open("PDDL/domain.pddl", "r") as f:
            real_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

            # Store real action model effects

            all_action_schema = " ".join(real_action_model)[" ".join(real_action_model).index(":action "):]
            # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
            action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if
                             el.strip() != ""]

            for schema in action_schema:
                op_name = schema.split()[1]
                op_params = [el for el in re.findall("\([^()]*\)", re.findall(":parameters.*:precondition", schema)[0])[0].strip()[1:-1].split()
                             if el.startswith("?")]

                all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

                cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)
                # cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                #                   if el not in [el.replace("(not", "").strip()[:-1] for el in cur_neg_effect]
                #                   and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

                for k in range(len(cur_neg_effect)):

                    for j,param in enumerate(op_params):
                            # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))

                # for k in range(len(cur_pos_effect)):
                #
                #     for j,param in enumerate(op_params):
                #             # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                #             cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                #             cur_pos_effect[k] = cur_pos_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                #             cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))




                real_action_eff_neg[op_name] = cur_neg_effect

    # tp_eff = 0
    # fp_eff = 0
    # fn_eff = 0
    ins_del = 0
    del_del = 0

    for key, value in real_action_eff_neg.items():
        for pred in value:
            if pred not in learned_action_eff_neg[key]:
                ins_del += 1

    for key, value in learned_action_eff_neg.items():
        for pred in value:
            if pred not in real_action_eff_neg[key]:
                del_del += 1

    return ins_del, del_del



def action_model_eff_neg_statistics_with_uncertain(uncert_neg_eff):

    real_action_eff_neg = defaultdict(list)

    learned_action_eff_neg = defaultdict(list)

    # Compute action model coverage and overfitting
    with open("PDDL/domain_learned.pddl", "r") as f:
        learned_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

        # Store learned action model effects

        all_action_schema = " ".join(learned_action_model)[" ".join(learned_action_model).index(":action "):]
        # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
        action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if el.strip() != ""]

        for schema in action_schema:
            op_name = schema.split()[1]
            all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

            cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)
            # cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
            #                   if el not in [el.replace("(not","").strip()[:-1] for el in cur_neg_effect]
            #                   and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

            learned_action_eff_neg[op_name] = cur_neg_effect + ["(not {})".format(el) for el in uncert_neg_eff[op_name]]




        with open("PDDL/domain.pddl", "r") as f:
            real_action_model = [el.lower() for el in f.read().split('\n') if el.strip() != ""]

            # Store real action model effects

            all_action_schema = " ".join(real_action_model)[" ".join(real_action_model).index(":action "):]
            # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
            action_schema = [el.strip() for el in re.findall("(?:(?!:action).)*", all_action_schema) if
                             el.strip() != ""]

            for schema in action_schema:
                op_name = schema.split()[1]
                op_params = [el for el in re.findall("\([^()]*\)", re.findall(":parameters.*:precondition", schema)[0])[0].strip()[1:-1].split()
                             if el.startswith("?")]

                all_eff = re.findall(":effect.*", schema)[0].strip()[:-1].strip()

                cur_neg_effect = re.findall("\(not[^)]*\)\)", all_eff)
                # cur_pos_effect = [el for el in re.findall("\([^()]*\)", all_eff)
                #                   if el not in [el.replace("(not", "").strip()[:-1] for el in cur_neg_effect]
                #                   and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

                for k in range(len(cur_neg_effect)):

                    for j,param in enumerate(op_params):
                            # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                            cur_neg_effect[k] = cur_neg_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))

                # for k in range(len(cur_pos_effect)):
                #
                #     for j,param in enumerate(op_params):
                #             # action_precond[p] = action_precond[p].replace(el, "?param_{}".format(action_params.index(el)+1))
                #             cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + " ", " ?param_{} ".format(j + 1))
                #             cur_pos_effect[k] = cur_pos_effect[k].replace("(" + param + " ", "(?param_{} ".format(j + 1))
                #             cur_pos_effect[k] = cur_pos_effect[k].replace(" " + param + ")", " ?param_{})".format(j + 1))




                real_action_eff_neg[op_name] = cur_neg_effect

    # tp_eff = 0
    # fp_eff = 0
    # fn_eff = 0
    ins_del = 0
    del_del = 0

    for key, value in real_action_eff_neg.items():
        for pred in value:
            if pred not in learned_action_eff_neg[key]:
                ins_del += 1

    for key, value in learned_action_eff_neg.items():
        for pred in value:
            if pred not in real_action_eff_neg[key]:
                del_del += 1

    return ins_del, del_del
