# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import re

class Simulator:

    def __init__(self):

        with open("PDDL/facts.pddl", "r") as f:
            data = [el.strip() for el in f.read().split("\n")]
            facts = re.findall(":init.*\(:goal", "".join(data))[0]
        current_state_pddl = sorted(re.findall("\([^()]*\)", facts))

        self.state = current_state_pddl

        pass


    def get_state(self):
        return sorted(self.state)


    def execute(self, action_label):

        executable = self.check_action_precondition(action_label)

        if executable:
            self.apply_action_effects(action_label)

        return self.state, executable


    def check_action_precondition(self, action_label):

        operator = action_label.split("(")[0]
        a_params = action_label.split("(")[1][:-1].split(",")

        with open("PDDL/domain.pddl", "r") as f:
            data = [el.lower().strip() for el in f.read().split("\n")]

            all_action_schema = " ".join(data)[" ".join(data).index(":action"):]
            # action_schema = re.findall("{}(.*?):effect".format(operator), " ".join(data))[0]
            action_schema = re.findall(":action {}(.*?):effect".format(operator), all_action_schema)[0]
            preconds = sorted(re.findall("\([^()]*\)", action_schema[action_schema.find("precondition"):]))

            # op_params = [el for i,el in enumerate(re.findall(":parameters(.*?):precondition", action_schema)[0].strip()[1:-1].split()) if i%3==0]

            op_params_row = re.findall(":parameters(.*?):precondition", action_schema)[0]

            # single_obj_count = 0
            op_params = []
            for el in [el for el in op_params_row.strip()[1:-1].split() if el.strip() != "-"]:
                if el.startswith("?"):
                    op_params.append(el)

            preconds = "++".join(preconds)

            for i, el in enumerate(op_params):
                preconds = preconds.replace(el + ")", a_params[i] + ")")
                preconds = preconds.replace(el + " ", a_params[i] + " ")

            preconds = preconds.split("++")

            if set(preconds).issubset(set(self.state)):
                return True

        return False


    def apply_action_effects(self, action_label):

        operator = action_label.split("(")[0]
        a_params = action_label.split("(")[1][:-1].split(",")

        with open("PDDL/domain.pddl", "r") as f:
            data = [el.lower().strip() for el in f.read().split("\n")]

        all_action_schema = " ".join(data)[" ".join(data).index(":action"):]
        action_schema = re.findall(":action {}(.*?)\(:action".format(operator), all_action_schema)

        if len(action_schema) > 0:
            action_schema = action_schema[0]
        # Searched action is the last written one in the domain file
        else:
            action_schema = re.findall(":action {}.*".format(operator), all_action_schema)[0]


        # if action_schema[action_schema.find("effect"):].find("(and") != -1:
        #     effects = action_schema[action_schema.find("effect"):].strip().replace("effect", "").replace("(and", "")[:-3].strip()
        # else:
        #     effects = action_schema[action_schema.find("effect"):].strip().replace("effect", "").replace("(and", "")[:-2].strip()


        # effects = action_schema[action_schema.find("effect"):].strip().replace("effect", "").replace("(and", "")[:-2].strip()
        effects = action_schema[action_schema.find("effect"):]


        # positive_effects = re.findall("\([^()]*\)", re.sub("\(not[^)]*\)\)", "", effects))
        # negative_effects = re.findall("\(not[^)]*\)\)", effects)
        positive_effects = re.findall("\([^()]*\)", re.sub("\(not[^)]*\)\)", "", effects))
        negative_effects = re.findall("\(not[^)]*\)\)", effects)

        # op_params = [el for i,el in enumerate(re.findall(":parameters(.*?):precondition", action_schema)[0].strip()[1:-1].split()) if i%3==0]
        op_params = [el for el in re.findall(":parameters(.*?):precondition", action_schema)[0].strip()[1:-1].split()
                     if el.strip().startswith("?")]

        positive_effects = "++".join(positive_effects)
        negative_effects = "++".join(negative_effects)

        for i, el in enumerate(op_params):
            positive_effects = positive_effects.replace(" " + el + " ", " " + a_params[i]+ " ")
            positive_effects = positive_effects.replace(" " + el + ")", " " + a_params[i]+ ")")
            negative_effects = negative_effects.replace(" " + el + " ", " " + a_params[i]+ " ")
            negative_effects = negative_effects.replace(" " + el + ")", " " + a_params[i]+ ")")

        positive_effects = positive_effects.split("++")
        negative_effects = negative_effects.split("++")
        negative_effects = [el.replace("(not", "").strip()[:-1] for el in negative_effects]

        self.state = [el for el in self.state if el not in negative_effects]
        self.state = self.state + [el for el in positive_effects]

        self.state = list(set(self.state))





