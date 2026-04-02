
cpdef compute_not_executable_actions_cy(
        list current_state,
        list action_list,
        dict action_preconditions,
        str failed_action_label):
    cdef:
        set not_exec_new = set()
        set pddl_state_set
        str operator
        Py_ssize_t i
        list new_action_list = []
        list new_idx_list = []


    operator = failed_action_label.split("(", 1)[0]

    pddl_state_set = set(current_state)

    if operator in action_preconditions:
        action_preconditions = {operator: action_preconditions[operator]}
    else:
        action_preconditions = dict()

    # for a, idx in zip(action_list, action_indexes):
    for a in action_list:
        op_part = a.split("(", 1)[0]
        if op_part == operator:
            if not check_action_precondition(
                pddl_state_set,
                a,
                action_preconditions
            ):
                not_exec_new.add(a)

    return not_exec_new


cpdef compute_all_not_executable_actions_cy(
        list current_state,
        list action_list,
        list action_indexes,
        dict action_preconditions):
    cdef:
        list operators = []
        set operators_set = set()
        set not_exec = set()
        set pddl_state_set
        Py_ssize_t i
        str label

    operators_set = {a.split("(", 1)[0] for a in action_list}

    pddl_state_set = set(current_state)

    for i in range(len(action_list)):
        if not check_action_precondition(
            pddl_state_set,
            action_list[i],
            action_preconditions
        ):
            not_exec.add(action_list[i])

    return not_exec


cpdef bint check_action_precondition(
        set pddl_state,
        str action_label,
        dict action_preconditions):

    cdef:
        str operator
        list preconds
        list params
        list final_preconds
        Py_ssize_t i, n
        int satisfied_neg, satisfied_pos
        str pred
        str ground

    operator = action_label.split("(", 1)[0]

    if operator not in action_preconditions:
        return True

    params = action_label[len(operator) + 1:len(action_label) - 1].split(",")

    for preconds in action_preconditions[operator]:

        # ground
        ground = "++".join(preconds)
        for i in range(len(params)):
            ground = ground.replace(f"param_{i+1}", params[i])

        final_preconds = ground.split("++")
        n = len(final_preconds)

        satisfied_neg = 0
        satisfied_pos = 0

        for pred in final_preconds:
            if pred.startswith("not"):
                if pred[4:len(pred) - 1] in pddl_state:
                    break
                satisfied_neg += 1
            else:
                if pred not in pddl_state:
                    break
                satisfied_pos += 1

        if satisfied_neg + satisfied_pos == n:
            return False

    return True
