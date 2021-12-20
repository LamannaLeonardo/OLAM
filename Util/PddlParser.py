import re

class PddlParser():

    def __init__(self):
        pass

    def update_pddl_facts(self, obs):

        facts_file = "PDDL/facts.pddl"

        # ground_atoms = sorted(["(" + re.sub(":[^,]*(,|\))"," ",str(el))[:-1].replace("("," ").replace(")", " ") + ")"
        #                        for el in obs.literals])

        ground_atoms = obs

        with open(facts_file, "r") as f:
            data = f.read().split("\n")

        with open(facts_file, "w") as f:

            start_index = None
            end_index = None

            for i in range(len(data)):
                line = data[i]

                if line.find(":init") != -1:
                    start_index = i

                if line.find(":goal") != -1:
                    end_index = i

            for i in range(len(data)):
                if i==start_index:
                    if not data[i].strip().startswith(")"):
                        f.write("\t(:init\n\t\t\t{}\n\t)\n".format("\n\t\t\t".join(ground_atoms)))
                    else:
                        f.write(")\n\t(:init\n\t\t\t{}\n\t)\n".format("\n\t\t\t".join(ground_atoms)))

                elif not start_index<i<end_index:
                    f.write(data[i] + "\n")
