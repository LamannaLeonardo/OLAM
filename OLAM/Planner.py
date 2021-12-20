import collections
import datetime
import heapq
import os
import subprocess
import re
import sys

import Configuration

PLAN_FOUND = False


def FF():
    """
    Compute the plan using FastForward planner, the pddl file (i.e. "domain.pddl" and "facts.pddl") must be contained
    into the "PDDL" folder.
    :return: a plan
    """

    # DEBUG
    start = datetime.datetime.now()

    bash_command = "Planners/FF/ff -o PDDL/domain.pddl -f PDDL/facts.pddl"

    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

    output, error = process.communicate()

    begin_index = -1
    end_index = -1
    result = str(output).split("\\n")

    for el in result:
        if el.__contains__("No plan will solve it"):
            return [], False

    for i in range(len(result)):
        if not result[i].find("step"):
            begin_index = i

        elif not result[i].find("time"):
            end_index = i-2

    plan = [result[i].split(":")[1].replace("\\r", "") for i in range(begin_index, end_index)]
    syntax_plan = []

    if len(plan) == 0:
        print('planner plan is empty')
        sys.exit()

    # DEBUG
    end = datetime.datetime.now()
    print("FF computational time: {}".format(end-start))

    for el in plan:
        tmp = el
        tmp = re.sub("[ ]", ",", tmp.strip())
        tmp = tmp.replace(",", "(", 1)
        tmp = tmp + ")"
        tmp = tmp[:tmp.index('(')].replace("-","_") + tmp[tmp.index('('):]
        syntax_plan.append(tmp)

    return syntax_plan, True




def Madagascar(domain, facts):
    """
    Compute the plan using Madagascar planner, the pddl file (i.e. "domain.pddl" and "facts.pddl") must be contained
    into the "PDDL" folder.
    :return: a plan
    """

    # DEBUG
    start = datetime.datetime.now()



    bash_command = "Planners/ADL2STRIPS/adl2strips -o PDDL/{} -f PDDL/{}".format(domain, facts)

    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    process.wait()



    bash_command = "Planners/Madagascar/M domain.pddl facts.pddl -o 'mad_out.txt' -P 0 -T 100 -S 50"

    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    process.wait()

        # ./Planners/FD/fast-downward.py PDDL/domain.pddl PDDL/facts.pddl --evaluator "hff=ff()" --evaluator "hcea=cea()" --search "lazy_greedy([hff,hcea],preferred=[hff,hcea])"


    process = subprocess.Popen(bash_command.replace("\"", "").split(), stdout=subprocess.PIPE)

    output, error = process.communicate()

    # DEBUG
    end = datetime.datetime.now()
    print("Madagascar computational time: {}".format(end-start))


    plan_found = False
    if str(output).lower().find("plan found") != -1:
        plan_found = True
    else:
        return None, plan_found

    # Get operators list
    operators = []
    with open("PDDL/domain_learned.pddl", "r") as f:
        data = f.read().split("\n")

        for i in range(len(data)):

            if data[i].find("action") != -1:
                operators.append(data[i].split()[1].strip())



    plan = []

    with open(os.path.join(os.getcwd(), 'mad_out.txt'), 'r') as file:

        data = [el for el in file.read().split("\n") if el.strip() != ""]

        data = [el.split(":")[1].replace("()", "").strip().lower()[:-2] for el in data]


        for i in range(len(data)):

            for op in operators:
                if data[i].find(op) != -1:
                    data[i] = data[i].replace(op + "-", op + "(")

                    action_sign = op + "("

                    action_obj = ""
                    for obj in data[i].split("(")[1].split("-"):
                        if obj.isnumeric() and len(action_obj) > 0:
                            action_obj += ("-" + obj)
                        elif len(action_obj) == 0:
                            action_obj += obj
                        else:
                            action_sign += (action_obj + ",")
                            action_obj = obj

                    action_sign += (action_obj + ")")

                    plan.append(action_sign)


    # os.remove(os.path.join(os.getcwd(), 'sas_plan'))

    return plan, plan_found


def FD():
    """
    Compute the plan using FastDownward planner, the pddl file (i.e. "domain.pddl" and "facts.pddl") must be contained
    into the "PDDL" folder. The plan is written in the output file "sas_plan"
    :return: a plan
    """

    # DEBUG
    start = datetime.datetime.now()

    bash_command = "./Planners/FD/fast-downward.py --overall-time-limit {}" \
                    " PDDL/domain_dummy.pddl PDDL/facts_dummy.pddl " \
                   "--evaluator \"hff=ff()\" " \
                   "--evaluator \"hcea=cea()\" " \
                   "--search \"lazy_greedy([hff,hcea],preferred=[hff,hcea])\""\
        .format(Configuration.PLANNER_TIME_LIMIT)

    process = subprocess.Popen(bash_command.replace("\"", "").split(), stdout=subprocess.PIPE)

    output, error = process.communicate()

    # DEBUG
    end = datetime.datetime.now()
    print("FD computational time: {}".format(end-start))

    syntax_plan = []

    if str(output).find("no solution") != -1:
        return None, False

    if str(output).lower().find("time limit has been reached") != -1 \
        or str(output).lower().find("translate exit code: 21") != -1 \
        or str(output).lower().find("translate exit code: -9") != -1:
        # transalte exit code: 21 means time limit reached in the translator
        return None, False

    with open(os.path.join(os.getcwd(), 'sas_plan'), 'r') as file:

        data = file.read().split('\n')

        data = list(filter(lambda row: row.find(";") == -1 and len(row) > 0, data))

        # Check plan exists
        # if len(data) == 0:
        #     return [], False

        for el in data:
            el = el[1:-1]
            params = el.split()
            # tmp = params[0].replace("-","_") + "("
            tmp = params[0] + "("

            tmp += ",".join(params[1:])

            tmp += ")"

            syntax_plan.append(tmp.upper())

    os.remove(os.path.join(os.getcwd(), 'sas_plan'))

    return syntax_plan, True



def FD_test():
    """
    Compute the plan using FastDownward planner, the pddl file (i.e. "domain_dummy.pddl" and "facts_dummy.pddl") must be contained
    into the "PDDL" folder. The plan is written in the output file "sas_plan"
    :return: a plan
    """

    # DEBUG
    start = datetime.datetime.now()

    bash_command = "./Planners/FD/fast-downward.py --overall-time-limit {}" \
                   " PDDL/domain_test.pddl PDDL/facts_test.pddl " \
                   "--evaluator \"hff=ff()\" " \
                   "--evaluator \"hcea=cea()\" " \
                   "--search \"lazy_greedy([hff,hcea],preferred=[hff,hcea])\""\
        .format(Configuration.PLANNER_TIME_LIMIT)

    process = subprocess.Popen(bash_command.replace("\"", "").split(), stdout=subprocess.PIPE)

    output, error = process.communicate()

    # DEBUG
    end = datetime.datetime.now()
    print("FD computational time: {}".format(end-start))

    syntax_plan = []

    if str(output).find("no solution") != -1:
        return None, False

    if str(output).lower().find("time limit has been reached") != -1 \
        or str(output).lower().find("translate exit code: 21") != -1 \
        or str(output).lower().find("translate exit code: -9") != -1:
        # transalte exit code: 21 means time limit reached in the translator
        return None, False

    with open(os.path.join(os.getcwd(), 'sas_plan'), 'r') as file:

        data = file.read().split('\n')

        data = list(filter(lambda row: row.find(";") == -1 and len(row) > 0, data))

        # Check plan exists
        # if len(data) == 0:
        #     return [], False

        for el in data:
            el = el[1:-1]
            params = el.split()
            # tmp = params[0].replace("-","_") + "("
            tmp = params[0] + "("

            tmp += ",".join(params[1:])

            tmp += ")"

            syntax_plan.append(tmp.upper())

    os.remove(os.path.join(os.getcwd(), 'sas_plan'))

    return syntax_plan, True



def FD_dummy():
    """
    Compute the plan using FastDownward planner, the pddl file (i.e. "domain_dummy.pddl" and "facts_dummy.pddl") must be contained
    into the "PDDL" folder. The plan is written in the output file "sas_plan"
    :return: a plan
    """

    # DEBUG
    start = datetime.datetime.now()

    bash_command = "./Planners/FD/fast-downward.py --overall-time-limit {}" \
                   " PDDL/domain_dummy.pddl PDDL/facts_dummy.pddl " \
                   "--evaluator \"hff=ff()\" " \
                   "--evaluator \"hcea=cea()\" " \
                   "--search \"lazy_greedy([hff,hcea],preferred=[hff,hcea])\""\
        .format(Configuration.PLANNER_TIME_LIMIT)

    process = subprocess.Popen(bash_command.replace("\"", "").split(), stdout=subprocess.PIPE)

    output, error = process.communicate()

    # DEBUG
    end = datetime.datetime.now()
    print("FD computational time: {}".format(end-start))

    syntax_plan = []

    if str(output).find("no solution") != -1:

        print("Plan computed: False Plan length: None")
        return None, False

    if str(output).lower().find("time limit has been reached") != -1 \
        or str(output).lower().find("translate exit code: 21") != -1 \
        or str(output).lower().find("translate exit code: -9") != -1:
        # transalte exit code: 21 means time limit reached in the translator
        print("Plan computed: False Plan length: None")
        return None, False

    with open(os.path.join(os.getcwd(), 'sas_plan'), 'r') as file:

        data = file.read().split('\n')

        data = list(filter(lambda row: row.find(";") == -1 and len(row) > 0, data))

        # Check plan exists
        # if len(data) == 0:
        #     return [], False

        for el in data:
            el = el[1:-1]
            params = el.split()
            # tmp = params[0].replace("-","_") + "("
            tmp = params[0] + "("

            tmp += ",".join(params[1:])

            tmp += ")"

            syntax_plan.append(tmp.upper())

    os.remove(os.path.join(os.getcwd(), 'sas_plan'))

    print("Plan computed: True Plan length: {}".format(len(syntax_plan)))

    return syntax_plan, True


def FD_dummy_goal():
    """
    Compute the plan using FastDownward planner, the pddl file (i.e. "domain_learned.pddl" and "facts_dummy.pddl") must
    be contained into the "PDDL" folder. The plan is written in the output file "sas_plan"
    :return: a plan
    """

    # DEBUG
    start = datetime.datetime.now()

    bash_command = "./Planners/FD/fast-downward.py --overall-time-limit {}" \
                   " PDDL/domain_learned.pddl PDDL/facts_dummy.pddl " \
                   "--evaluator \"hff=ff()\" " \
                   "--evaluator \"hcea=cea()\" " \
                   "--search \"lazy_greedy([hff,hcea],preferred=[hff,hcea])\""\
        .format(Configuration.PLANNER_TIME_LIMIT)

    process = subprocess.Popen(bash_command.replace("\"", "").split(), stdout=subprocess.PIPE)

    output, error = process.communicate()

    # DEBUG
    end = datetime.datetime.now()
    print("FD computational time: {}".format(end-start))

    syntax_plan = []

    if str(output).find("no solution") != -1:
        return None, False

    if str(output).lower().find("time limit has been reached") != -1 \
        or str(output).lower().find("translate exit code: 21") != -1 \
        or str(output).lower().find("translate exit code: -9") != -1:
        # transalte exit code: 21 means time limit reached in the translator
        return None, False

    with open(os.path.join(os.getcwd(), 'sas_plan'), 'r') as file:

        data = file.read().split('\n')

        data = list(filter(lambda row: row.find(";") == -1 and len(row) > 0, data))

        # Check plan exists
        # if len(data) == 0:
        #     return [], False

        for el in data:
            el = el[1:-1]
            params = el.split()
            # tmp = params[0].replace("-","_") + "("
            tmp = params[0] + "("

            tmp += ",".join(params[1:])

            tmp += ")"

            syntax_plan.append(tmp.upper())

    os.remove(os.path.join(os.getcwd(), 'sas_plan'))

    return syntax_plan, True


def FD_dummy_real_goal():
    """
    Compute the plan using FastDownward planner, the pddl file (i.e. "domain_dummy.pddl" and "facts.pddl") must be contained
    into the "PDDL" folder. The plan is written in the output file "sas_plan"
    :return: a plan
    """

    # DEBUG
    start = datetime.datetime.now()

    bash_command = "./Planners/FD/fast-downward.py --overall-time-limit {}" \
                   " PDDL/domain_dummy.pddl PDDL/facts.pddl " \
                   "--evaluator \"hff=ff()\" " \
                   "--evaluator \"hcea=cea()\" " \
                   "--search \"lazy_greedy([hff,hcea],preferred=[hff,hcea])\""\
        .format(Configuration.PLANNER_TIME_LIMIT)

    process = subprocess.Popen(bash_command.replace("\"", "").split(), stdout=subprocess.PIPE)

    output, error = process.communicate()

    # DEBUG
    end = datetime.datetime.now()
    print("FD computational time: {}".format(end-start))

    syntax_plan = []

    if str(output).find("no solution") != -1:
        return None, False

    if str(output).lower().find("time limit has been reached") != -1 \
        or str(output).lower().find("translate exit code: 21") != -1 \
        or str(output).lower().find("translate exit code: -9") != -1:
        # translate exit code: 21 means time limit reached in the translator
        return None, False

    with open(os.path.join(os.getcwd(), 'sas_plan'), 'r') as file:

        data = file.read().split('\n')

        data = list(filter(lambda row: row.find(";") == -1 and len(row) > 0, data))

        # Check plan exists
        # if len(data) == 0:
        #     return [], False

        for el in data:
            el = el[1:-1]
            params = el.split()
            # tmp = params[0].replace("-","_") + "("
            tmp = params[0] + "("

            tmp += ",".join(params[1:])

            tmp += ")"

            syntax_plan.append(tmp.upper())

    os.remove(os.path.join(os.getcwd(), 'sas_plan'))

    return syntax_plan, True




def BFWS():
    """
    Compute the plan using bfws planner, the pddl file (i.e. "domain.pddl" and "facts.pddl") must be contained
    into the "PDDL" folder. The plan is written in the output file "plan.ipc"
    :return: a plan
    """

    # DEBUG
    start = datetime.datetime.now()

    bash_command = "./Planners/bfws-fd-version/bfws.py PDDL/domain_learned.pddl PDDL/facts.pddl k-BFWS"

    process = subprocess.Popen(bash_command.replace("\"", "").split(), stdout=subprocess.PIPE)

    output, error = process.communicate()

    # DEBUG
    end = datetime.datetime.now()
    print("BFWS computational time: {}".format(end-start))

    syntax_plan = []

    with open(os.path.join(os.getcwd(), 'plan.ipc'), 'r') as file:

        data = file.read().split('\n')

        data = list(filter(lambda row: row.find(";") == -1 and len(row) > 0, data))

        # Check plan exists
        # if len(data) == 0:
        #     return [], False

        for el in data:
            el = el[1:-1]
            params = el.split()
            tmp = params[0].replace("-","_") + "("

            tmp += ",".join(params[1:])

            tmp += ")"

            syntax_plan.append(tmp.upper())

    return syntax_plan, True
