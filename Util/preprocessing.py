# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import re
import shutil
from collections import defaultdict
from shutil import copyfile

import Configuration


def preprocess(simulator_domain):

    if not os.path.exists("PDDL"):
        os.mkdir("PDDL")

    assert os.path.exists(os.path.join(Configuration.ROOT_BENCHMARKS_DIR, simulator_domain + ".pddl")), "PDDL domain " \
                                        " file must be in \"Analysis/Benchmarks\" directory and must have the same name" \
                                        " of the domain benchmark directory."

    # Get domain functions
    with open(os.path.join(Configuration.ROOT_BENCHMARKS_DIR, simulator_domain + ".pddl"), "r") as f:
        data = [el.lower() for el in f.read().split("\n") if not el.strip().startswith(";")]

    functions = []
    if len(re.findall(":functions(.*?):action", "".join(data))) > 0:
        functions = [el.strip()[1:-1].split()[0] for el in
                     re.findall("\([^\(\)]*\)", re.findall(":functions(.*?):action", "".join(data))[0])]

    # Copy original domain to working file
    preprocess_domain_learned(simulator_domain)
    preprocess_domain_dummy()

    # Copy original problem instance to working file
    copyfile(Configuration.INSTANCE_DATA_PATH_PDDL, "PDDL/facts.pddl")

    # Remove action cost predicate from facts file
    with open("PDDL/facts.pddl", "r") as f:
        data = f.read().split("\n")

    with open("PDDL/facts.pddl", "w") as f:
        for i in range(len(data)):
            # if data[i].find("total-cost") != -1:
            #     data[i] = ""

            if data[i].strip().startswith(";"):
                data[i] = ""

            else:
                for fun in functions:
                    if data[i].find(fun) != -1:
                        data[i] = ""

        [f.write(el.lower() + "\n") for el in data if el.strip() != ""]


    # Format facts file
    with open("PDDL/facts.pddl", "r") as f:
        data = [el.strip() for el in f.read().split("\n")]

    with open("PDDL/facts.pddl", "w") as f:

        for j in range(len(data)):
            if data[j].find(":goal") != -1:
                if data[j].strip().startswith(")"):
                    data[j] = ")\n{}".format(data[j].replace(")","",1))
                break

        [f.write(el.lower() + "\n") for el in data if el.strip() != ""]


def preprocess_domain_learned(simulator_domain):

    # if os.path.exists("PDDL/domain_input.pddl"):
    #     shutil.copyfile("PDDL/domain_input.pddl", "PDDL/domain_learned.pddl")
    #     return

    # Copy original typed domain into working domain file
    copyfile(os.path.join(Configuration.ROOT_BENCHMARKS_DIR, simulator_domain + ".pddl"), Configuration.DOMAIN_FILE_NAME)

    # Translate original domain into parameterized learned domain
    learned_domain_path = os.path.join("PDDL", "domain_learned.pddl")
    copyfile(os.path.join(Configuration.ROOT_BENCHMARKS_DIR, simulator_domain + ".pddl"), learned_domain_path)

    with open(learned_domain_path, "r") as f:
        data = [el.lower() for el in f.read().split("\n") if not el.strip().startswith(";")]


    # Remove domain functions
    for i in range(len(data)):

        if data[i].find(":action-costs") != -1:
            data[i] = data[i].replace(":action-costs", "")

        if data[i].find(":functions") != -1:

            for j in range(i, len(data)):

                if data[j].find(":action") != -1:
                    break
                else:
                    data[j] = ""


    with open(learned_domain_path, "w") as f:

        all_action_schema = []
        action_indices = []

        for i in range(len(data)):
            row = data[i]

            if row.find(":action ") != -1:
                action_indices.append(i)


        for i in range(len(action_indices)):

            action_index = action_indices[i]

            if action_index != action_indices[-1]:

                action_schema = "".join(data[action_index:action_indices[i+1]])

                # action_schema = re.sub(':precondition.*:effect', ':precondition (and\n):effect', action_schema)
                action_schema = re.sub(':precondition.*', ':precondition (and\n):effect (and ))', action_schema)

                action_schema = re.sub(' +|\t', ' ', action_schema).replace(":", "\n:").replace("\n:", ":", 1)

            else:

                action_schema = "".join(data[action_index:])


                # action_schema = re.sub(':precondition.*:effect', ':precondition (and\n):effect', action_schema)
                action_schema = re.sub(':precondition.*', ':precondition (and\n):effect (and ))', action_schema)

                # action_schema = re.sub(' +|\t', ' ', action_schema).replace(":", "\n:").replace("\n:", ":", 1)[:-1] + "\n)"

                # action_schema = re.sub(' +|\t', ' ', action_schema).replace(":", "\n:").replace("\n:", ":", 1)[:-1]
                action_schema = re.sub(' +|\t', ' ', action_schema).replace(":", "\n:").replace("\n:", ":", 1)


            params = [el for i, el in enumerate(re.findall("\(.*\)", action_schema.split("\n")[1])[0][1:-1].split())
                      if el.startswith("?")]

            for k, param in enumerate(params):
                action_schema = action_schema.replace("({} ".format(param), "(?param_{} ".format(k + 1))
                action_schema = action_schema.replace(" {} ".format(param), " ?param_{} ".format(k + 1))
                action_schema = action_schema.replace(" {})".format(param), " ?param_{})".format(k + 1))

            all_action_schema.append(action_schema)

        for i in range(len(data)):
            if data[i].find(":action ") != -1:
                break
            f.write("\n" + data[i])

        [f.write("\n\n{}".format(action_schema)) for action_schema in all_action_schema]
        f.write("\n)")


    # Add input domain effects
    if os.path.exists("PDDL/domain_input.pddl"):

        all_action_effects = defaultdict(list)

        with open("PDDL/domain_input.pddl", "r") as f:
            data = f.read().split("\n")

            for i in range(len(data)):

                if data[i].find(":action ") != -1:

                    action_effects = data[i+5]

                    all_action_effects[data[i].strip().split()[1]] = action_effects


        with open("PDDL/domain_learned.pddl", "r") as f:

            data = f.read().split("\n")

            for i in range(len(data)):

                if data[i].find(":action ") != -1:

                    if all_action_effects[data[i].strip().split()[1]].replace("(and ","").replace(")","").strip() != "":
                        data[i+4] = all_action_effects[data[i].strip().split()[1]]
                    else:
                        data[i+4] = ":effect (and )"


            with open("PDDL/domain_learned.pddl", "w") as f:
                [f.write(el.lower()+ "\n") for el in data]







def preprocess_domain_dummy():

    # Compute initial domain dummy (i.e., domain with only learned operators)
    copyfile("PDDL/domain_learned.pddl", os.path.join("PDDL/domain_dummy.pddl"))

    with open("PDDL/domain_dummy.pddl", 'r') as f:
        data = [el.lower() for el in f.read().split("\n") if el.strip() != ""]

        removed_rows = []

        for m in range(len(data)):

            row = data[m]

            # if row.lower().strip().startswith("(:action"):
            if row.lower().strip().find(":action ") != -1:

                if data[m + 2].strip().startswith(":precondition") and data[m + 3].strip() == ")":
                    removed_rows.append(m)

                    for n in range(m + 1, len(data) - 1):
                        # if data[n].strip().startswith("(:action"):
                        if data[n].strip().find(":action ") != -1:
                            break
                        removed_rows.append(n)

        dummy_domain = [data[i] for i in range(len(data)) if i not in removed_rows]

    with open("PDDL/domain_dummy.pddl", 'w') as f:
        [f.write(el.lower() + "\n") for el in dummy_domain]
