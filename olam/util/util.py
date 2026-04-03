import itertools
import re

from unified_planning.io import PDDLReader, PDDLWriter
from unified_planning.model import Fluent, Object, Problem
from unified_planning.plans import ActionInstance
from unified_planning.shortcuts import BoolType

from olam.modeling.symbolic_observation import SymbolicObservation


def empty_domain(domain_path: str, empty_domain_path: str = 'empty.pddl'):
    pddl_domain = PDDLReader().parse_problem(domain_path)

    # Loop through all actions and remove preconditions and effects
    for action in pddl_domain.actions:
        action.clear_preconditions()
        action.clear_effects()

    domain_str = PDDLWriter(pddl_domain).get_domain()
    pattern = re.compile(
        r"(:action[\s\S]*?:parameters\s*\([^)]*\))\)",
        re.MULTILINE
    )

    replacement = r"\1\n  :precondition (and )\n  :effect (and ))\n"

    # TODO: open issue in up
    domain_str = domain_str.replace(f"(domain {pddl_domain.name}-domain)",
                                    f"(domain {pddl_domain.name})")

    with open(empty_domain_path, 'w') as f:
        f.write(pattern.sub(replacement, domain_str))

    return empty_domain_path


def ground_lifted_atoms(action_instance: ActionInstance, lifted_atoms):
    grounded_atoms = set()
    for atom in lifted_atoms:
        subs_dict = {}
        for arg in atom.args:
            for n, param in enumerate(action_instance.action.parameters):
                if param.name == str(arg):
                    subs_dict[arg] = action_instance.actual_parameters[n]

        grounded_atoms.add(str(atom.substitute(subs_dict)))
    return grounded_atoms


def get_op_relevant_predicates(op, domain):
    """
    Get the domain predicates whose set of parameter types is a subset of the operator
    parameter types.
    """
    relevant_predicates = []

    for predicate in domain.fluents:
        sig = list(predicate.signature)

        # 0-arity fluent
        if len(sig) == 0:
            relevant_predicates.append(predicate())
            continue

        candidate_indices_per_arg = []
        feasible = True

        for pred_arg in sig:
            pred_type = pred_arg.type
            candidates = [
                i
                for i, op_param in enumerate(op.parameters)
                if pred_type.is_compatible(op_param.type)
            ]
            if not candidates:
                feasible = False
                break
            candidate_indices_per_arg.append(candidates)

        if not feasible:
            continue

        # Generate all consistent bindings
        for comb in itertools.product(*candidate_indices_per_arg):
            vars_for_pred = [op.parameters[i] for i in comb]
            relevant_predicates.append(predicate(*vars_for_pred))

    return sorted(relevant_predicates, key=lambda f: (f.fluent().name, str(f)))


def del_numeric_fluents(problem: Problem) -> Problem:
    new_p = Problem(problem.name)

    # Objects
    for o in problem.all_objects:
        new_p.add_object(o)

    # Fluents
    fluent_map = {}
    for f in problem.fluents:
        if f.type.is_bool_type():
            new_p.add_fluent(f)
            fluent_map[f] = f

    # Actions
    for a in problem.actions:
        new_p.add_action(a)

    # Initial values
    for fexp, val in problem.initial_values.items():
        if fexp.fluent() in fluent_map:
            new_p.set_initial_value(fexp, val)

    # Goals
    for g in problem.goals:
        new_p.add_goal(g)

    assert not any(not f.type.is_bool_type() for f in new_p.fluents), (
        "Numeric fluents still present!"
    )

    return new_p


def build_up_problem(
    domain: Problem, all_objects: list[Object], state: SymbolicObservation, goal
) -> Problem:
    problem = domain.clone()

    # Set objects
    problem.add_objects(all_objects)

    # Set initial state
    for f, value in state.fluents.items():
        fluent = problem.fluent(f.fluent().name)  # use problem fluent reference
        args = [problem.object(str(o)) for o in f.args]  # use problem object reference
        problem.set_initial_value(fluent(*args), value)

    # Adding at least one effect to every action
    dummy = Fluent("dummy", BoolType())
    problem.add_fluent(dummy, default_initial_value=False)
    for action in problem.actions:
        if len(action.effects) == 0:
            action.add_effect(dummy, True)

    # Set goal
    problem.clear_goals()
    problem.add_goal(goal)

    return problem
