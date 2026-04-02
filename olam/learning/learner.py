import itertools
import logging
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, ClassVar

from unified_planning.model import (
    Fluent,
    FNode,
    Object,
    OperatorKind,
    Problem,
)
from unified_planning.shortcuts import (
    BoolType,
    Not,
    OneshotPlanner,
)

from olam.learning.goal_generator import GoalGenerator
from olam.modeling.symbolic_observation import SymbolicObservation
from olam.util.util import build_up_problem


@dataclass
class Learner:
    """
    Args:
        domain (Problem): The unified planning domain
        all_objects (Any): A list of unified planning problem objects
        max_length (int): The maximum length of n-combinations of atoms to be considered
                         when generating goals for learning preconditions or effects.
                         For example, when max_n = 3, the agent plans to reach learning
                         states for refining preconditions where 1 or 2 or 3
                         preconditions are satisfied.
        max_goals (int): Maximum number of disjunctions in a goal formula
    """

    domain: Problem
    all_objects: Any
    max_length: int = 8
    max_subproblems: int = 5
    max_goals: int = 10000
    GOAL_PRECS: ClassVar[str] = "Learning preconditions"
    GOAL_ADD: ClassVar[str] = "Learning additive"
    GOAL_DEL: ClassVar[str] = "Learning delete"
    goal_kinds: ClassVar[list[str]] = [GOAL_PRECS, GOAL_ADD, GOAL_DEL]

    def __post_init__(self):
        self.goal_gen = GoalGenerator(self.domain,
                                      self.all_objects,
                                      max_length=self.max_length,
                                      max_goals=self.max_goals)

    def plan_for_operator(
        self,
        state: SymbolicObservation,
        operator,
        uncertain_positive_effects,
        uncertain_negative_effects,
        pre_bottom,
        pre_bot_ambiguous,
        executed_action_ambiguous,
        problem,
        invalid_actions,
        planner_cfg,
    ):

        # iterate over goals for learning: preconditions, additive, and delete effects
        for goal_kind in self.goal_kinds:

            # limit the maximum number of uncertain preconditions and effects to be
            # considered in each goal conjunction, based on the actual number of
            # uncertain preconditions and effects
            max_length = self.max_length
            if goal_kind == self.GOAL_PRECS:
                max_length = min(max_length, len(operator.preconditions))
            if goal_kind == self.GOAL_ADD:
                max_length = min(max_length, len(uncertain_positive_effects))
            if goal_kind == self.GOAL_DEL:
                max_length = min(max_length, len(uncertain_negative_effects))

            for length in range(1, max_length + 1):
                logging.debug(f"{goal_kind} of {operator.name} with length {length}")
                ###################################################################
                #  1: generate the goal for either learning preconditions or      #
                #     additive effects, or delete effects.                        #
                ###################################################################
                if goal_kind == self.GOAL_PRECS:
                    goal = self.goal_gen.get_goal_for_precs(
                        operator, length, pre_bottom, pre_bot_ambiguous, problem
                    )
                elif goal_kind == self.GOAL_ADD:
                    goal = self.goal_gen.get_goal_for_add(
                        operator,
                        length,
                        uncertain_positive_effects,
                        executed_action_ambiguous,
                        problem,
                        state,
                    )
                elif goal_kind == self.GOAL_DEL:
                    goal = self.goal_gen.get_goal_for_del(
                        operator,
                        length,
                        uncertain_negative_effects,
                        executed_action_ambiguous,
                        problem,
                        state,
                    )
                else:
                    raise NotImplementedError

                if goal.is_false():
                    length += 1
                    continue

                # If some object types are ambiguous (i.e., supertypes), get all
                # combination of object subtypes.
                # For example, if vehicle0 is an object of type "vechicle", and
                # "car" and "truck" are subtypes of "vehicle", then there is an
                # hypothesis 1 where vehicle0 is an object of type "car", and
                # an hypothesis 2 where vehicle0 is an object of type "truck"
                objs_hypothesis = self.get_objects_subtypes(goal)

                ###################################################################
                #  2: plan to achieve the generated goal                          #
                ###################################################################
                plan = None
                if len(objs_hypothesis) == 0:  # no ambiguous object (super)types
                    # build the planning problem
                    problem = build_up_problem(
                        self.domain, self.all_objects, state, goal
                    )
                    # solve the planning problem
                    with OneshotPlanner(**planner_cfg) as planner:
                        timeout = planner_cfg["params"][
                            "fast_downward_search_time_limit"
                        ]
                        result = planner.solve(problem, timeout=int(timeout[:-1]))

                        logging.debug(
                            f"Planning exited with status: {result.status.name}"
                        )
                        assert not result.status.name == "INTERNAL_ERROR"

                        plan = result.plan

                else:
                    # Generate a planning problem for every possible combination of
                    # ambiguous object (sub)types
                    possible_obj_types = list(
                        itertools.product(*objs_hypothesis.values())
                    )
                    possible_obj_types = random.choices(
                        possible_obj_types, k=self.max_subproblems
                    )
                    for obj_types in sorted(
                        list(set(possible_obj_types)), key=lambda x: str(x)
                    ):
                        # objects with known subtypes
                        objects = [
                            Object(o.name, o.type)
                            for o in self.all_objects
                            if o not in objs_hypothesis
                        ]
                        # objects with hypothesized subtypes
                        objects += [
                            Object(o.name, subtype)
                            for o, subtype in zip(
                                objs_hypothesis, obj_types, strict=True
                            )
                        ]

                        # create object references mapping for unified planning...
                        objs_map = dict()
                        goal_objs = [
                            problem.object(o_id)
                            for o_id in set(re.findall(r"==\s*([^)]+)\)", str(goal)))
                            if not o_id.startswith("param_")
                        ]
                        for o in objects:
                            for goal_obj in goal_objs:
                                if str(goal_obj) == str(o):
                                    objs_map[goal_obj] = o

                        # build the planning problem
                        possible_problem = build_up_problem(
                            self.domain, objects, state, goal.substitute(objs_map)
                        )

                        # prevent the plan from containing invalid actions, i.e.,
                        # actions involving invalid objects, which may be generated
                        # when planning with "wrong" hypothesized object subtypes.
                        # For example, if vehicle0 is hypothesized to be an object of
                        # subtype "truck", but its actually of type "car", then the
                        # action refuel-truck(vehicle0) would be invalid. In this case,
                        # if the agent already failed to execute such an action despite
                        # all preconditions being satisfied, then it learned such an
                        # action is invalid, and can prevent the plan from containing it
                        for possible_operator in possible_problem.actions:
                            forbid_pred = Fluent(
                                f"forbidden_{possible_operator.name}",
                                BoolType(),
                                possible_operator.parameters,
                            )
                            possible_problem.add_fluent(forbid_pred)
                            possible_operator.add_precondition(
                                Not(forbid_pred(*possible_operator.parameters))
                            )

                            for action in invalid_actions:
                                if action.action.name == possible_operator.name:
                                    args = [
                                        possible_problem.object(str(o))
                                        for o in action.actual_parameters
                                    ]

                                    # Check if the hypothesized object (sub)type is
                                    # compatible with the (invalid) action signature,
                                    # otherwise the action is already invalid in the
                                    # planning problem with hypothesized object subtypes
                                    if all(
                                        [
                                            o.type.is_compatible(p.type)
                                            for o, p in zip(
                                                args, forbid_pred.signature, strict=True
                                            )
                                        ]
                                    ):
                                        possible_problem.set_initial_value(
                                            forbid_pred(*args), True
                                        )

                        # solve the planning problem
                        with OneshotPlanner(**planner_cfg) as planner:
                            timeout = planner_cfg["params"][
                                "fast_downward_search_time_limit"
                            ]
                            result = planner.solve(
                                possible_problem, timeout=int(timeout[:-1])
                            )

                            logging.debug(
                                f"Planning exited with status: {result.status.name}"
                            )
                            assert not result.status.name == "INTERNAL_ERROR"

                            plan = result.plan

                        if plan is not None:
                            break

                assert plan is None or len(plan.actions) > 0

                # Store unsolvable (and positive) goals
                if plan is None and goal_kind in [self.GOAL_PRECS, self.GOAL_DEL]:
                    goal_conjunctions = list()
                    for g in goal.args:
                        if g.args[0].node_type == OperatorKind.AND:
                            for subgoal in g.args:
                                goal_conjunctions.append(
                                    [
                                        str(subsubgoal)
                                        for subsubgoal in subgoal.args
                                        if subsubgoal.node_type
                                        == OperatorKind.FLUENT_EXP
                                    ]
                                )
                        else:
                            goal_conjunctions.append(
                                [
                                    str(subgoal)
                                    for subgoal in g.args
                                    if subgoal.node_type == OperatorKind.FLUENT_EXP
                                ]
                            )
                    goal_conjunctions = {
                        frozenset(conj) for conj in goal_conjunctions if len(conj) > 0
                    }

                    self.goal_gen.unsolvable_goals[operator.name] |= goal_conjunctions

                    # Debugging
                    precs_str = {str(p) for p in operator.preconditions}
                    uncert_neg_str = {
                        str(e) for e in uncertain_negative_effects[operator.name]
                    }
                    assert all(
                        set(g).issubset(precs_str | uncert_neg_str)
                        for g in goal_conjunctions
                    ), (
                        "Goal for learning preconditions and uncertain negative "
                        "effects should involve only atoms that are resp. "
                        "preconditions and/or uncertain negative effects."
                    )

                if plan is not None:
                    return plan

        return None

    def get_objects_subtypes(self, goal: FNode):

        if len(goal.args) > 1:
            goal_vars = {v for g in goal.args for v in g.variables()}
        else:
            goal_vars = goal.variables()
        goal_types = {v.type for v in goal_vars}

        objs_hypothesis = []
        for v in goal_vars:
            objs_hypothesis.append(
                {
                    o: v.type
                    for o in self.all_objects
                    if o.type.is_compatible(v.type) and o.type != v.type
                }
            )
        objs_hypothesis = {
            o: {
                t
                for t in self.domain.user_types
                if o.type.is_compatible(t) and o.type != t
            }
            for o in self.all_objects
        }
        objs_hypothesis = {
            obj_id: subtypes.intersection(goal_types)
            for obj_id, subtypes in objs_hypothesis.items()
            if subtypes.intersection(goal_types)
        }

        return OrderedDict(objs_hypothesis)
