import logging
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from unified_planning.model import (
    FNode,
    Problem,
    Variable,
)
from unified_planning.shortcuts import (
    FALSE,
    TRUE,
    And,
    Equals,
    Exists,
    Not,
    Or,
)


@dataclass
class GoalGenerator:
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
    max_goals: int = 10000
    unsolvable_goals: dict[Any, set[Any]] = field(
        default_factory=lambda: defaultdict(set)
    )

    def get_goal_for_precs(
        self, operator, n, pre_bottom, pre_bot_ambiguous, problem
    ) -> FNode:

        pre_bottom = pre_bottom[operator.name]
        pre_bot_ambiguous = pre_bot_ambiguous[operator.name]

        precs = {str(p) for p in operator.preconditions}

        certain_precs = {str(list(p)[0]) for p in pre_bottom if len(p) == 1}
        unsatisfiable_precs = {
            str(list(p)[0]) for p in self.unsolvable_goals[operator.name] if len(p) == 1
        }
        goal_precs = precs - unsatisfiable_precs - certain_precs

        if n == len(goal_precs):
            return Or(*[])

        P_plus_comb = [
            p_comb
            for p_comb in combinations(goal_precs, n)
            if not np.any(
                [
                    p_bot.issubset(precs - certain_precs - set(p_comb))
                    for p_bot in pre_bottom
                ]
            )
            and not np.any(
                [
                    unsolvable.issubset(set(p_comb) | certain_precs)
                    for unsolvable in self.unsolvable_goals[operator.name]
                ]
            )
        ]

        if len(P_plus_comb) == 0:
            return Or(*[])

        # Add object conditions for precondition combinations that are already in
        # pre_bot, but with ambiguous object types
        object_conditions = []
        possible_args = [
            {str(o) for o in problem.all_objects if o.type.is_compatible(p.type)}
            for p in operator.parameters
        ]
        unfeasible_goal_ids = []
        for k, p_plus_comb in enumerate(P_plus_comb):
            all_object_conditions = [TRUE()]
            impossible_args = [set() for _ in operator.parameters]

            for cond, P_bot in pre_bot_ambiguous.items():
                if k in unfeasible_goal_ids:
                    break

                for p_bot in P_bot:
                    if k in unfeasible_goal_ids:
                        break

                    if p_bot.issubset(precs - certain_precs - set(p_plus_comb)):
                        object_condition = set()
                        for single_cond in cond:
                            param_idx = int(single_cond[0].split("_")[-1]) - 1
                            obj_id = single_cond[1]
                            param = operator.parameters[param_idx]
                            obj = problem.object(obj_id)

                            object_condition.add(Equals(param, obj))

                            impossible_args[param_idx].add(obj_id)

                            if any(
                                len(x - y) == 0
                                for x, y in zip(
                                    possible_args, impossible_args, strict=True
                                )
                            ):
                                unfeasible_goal_ids.append(k)
                                break

                        object_condition = Not(And(*object_condition))

                        all_object_conditions.append(object_condition)

                        break

            object_conditions.append(And(*all_object_conditions))

        assert len(object_conditions) == len(P_plus_comb)

        up_certain_precs = [
            p for p in operator.preconditions if str(p) in certain_precs
        ]
        up_P_plus_comb = [
            [p for p in operator.preconditions if str(p) in p_plus_comb]
            for p_plus_comb in P_plus_comb
        ]
        up_P_minus_comb = [
            [
                Not(p)
                for p in operator.preconditions
                if p not in p_plus_comb and p not in up_certain_precs
            ]
            for p_plus_comb in up_P_plus_comb
        ]

        goals = [
            And(*[P_plus_tuple, up_certain_precs, object_cond, Or(*P_minus_tuple)])
            for P_plus_tuple, P_minus_tuple, object_cond in zip(
                up_P_plus_comb, up_P_minus_comb, object_conditions, strict=True
            )
        ]

        # Filter out unfeasible goals
        goals = [g for k, g in enumerate(goals) if k not in unfeasible_goal_ids]
        if len(goals) == 0:
            processed_goals = [FALSE()]
        else:
            vars = [
                Variable(f"param_{i + 1}", p.type)
                for i, p in enumerate(operator.parameters)
            ]
            subs_map = {p: v for p, v in zip(operator.parameters, vars, strict=True)}
            processed_goals = [Exists(g.substitute(subs_map), *vars) for g in goals]
        # Instead of returning one mega Or, return many smaller goals
        goal = Or(*processed_goals)

        return goal

    def get_goal_for_add(
        self,
        operator,
        length,
        uncertain_positive_effects,
        executed_action_ambiguous,
        problem,
        state,
    ) -> FNode:

        goals = set()

        precs = {str(p) for p in operator.preconditions}

        eff_plus_possible = uncertain_positive_effects[operator.name]

        if not eff_plus_possible:
            return FALSE()

        unsatisfiable_for_precs = any(
            g.issubset(precs) for g in self.unsolvable_goals[operator.name]
        )
        if unsatisfiable_for_precs:
            logging.debug(
                "Goal unfeasible (for positive effects) because all "
                "preconditions cannot be satisfied"
            )
            return FALSE()

        # Get combination of action objects that have already been executed, and with
        # ambiguous types
        operator_params = [p for p in operator.parameters]
        object_conditions = {TRUE()}
        for action in executed_action_ambiguous[str(state)]:
            action_op = action.split("(")[0]
            if operator.name == action_op:
                action_objs = [
                    o.strip()
                    for o in action.split("(")[1][:-1].split(",")
                    if o.strip() != ""
                ]
                action_objs = [problem.object(o) for o in action_objs]

                object_condition = set()
                for param, obj in zip(operator_params, action_objs, strict=True):
                    object_condition.add(Equals(param, obj))
                object_conditions.add(Not(And(*object_condition)))

        for E_plus_tuple in combinations(eff_plus_possible, length):
            E_plus = set(E_plus_tuple)

            if any(el in operator.preconditions for el in E_plus):
                continue

            g = And(
                *(
                    [Not(ep) for ep in E_plus]
                    + list(operator.preconditions)
                    + list(object_conditions)
                )
            )

            if g not in goals:
                goals.add(g)

            if len(goals) > self.max_goals:
                logging.debug(f"Number of generated goals exceeds the maximum "
                              f"limit {self.max_goals}. Some goals are being "
                              f"discarded.")
                break

        if len(goals) == 0:
            processed_goals = [FALSE()]
        else:
            vars = [
                Variable(f"param_{i + 1}", p.type)
                for i, p in enumerate(operator.parameters)
            ]
            subs_map = {p: v for p, v in zip(operator.parameters, vars, strict=True)}

            # TODO: use only necessary inequalities for disambiguating uncertain effects
            inequalities = []
            for v1, v2 in combinations(vars, 2):
                if v1.type.is_compatible(v2.type) or v2.type.is_compatible(v1.type):
                    inequalities.append(Not(Equals(v1, v2)))
            inequalities = And(*inequalities) if inequalities else TRUE()

            processed_goals = [
                Exists(And(g.substitute(subs_map), And(inequalities)), *vars)
                for g in goals
            ]

        return Or(*processed_goals)

    def get_goal_for_del(
        self,
        operator,
        length,
        uncertain_negative_effects,
        executed_action_ambiguous,
        problem,
        state,
    ) -> FNode:

        goals = set()

        eff_minus_possible = uncertain_negative_effects[operator.name]

        precs = {str(p) for p in operator.preconditions}

        unsatisfiable_eff_minus = {
            str(list(p - set(precs))[0])
            for p in self.unsolvable_goals[operator.name]
            if len(p - set(precs)) == 1
        }

        goal_eff_minus = eff_minus_possible - unsatisfiable_eff_minus

        if not goal_eff_minus:
            return FALSE()

        unsatisfiable_for_precs = any(
            g.issubset(precs) for g in self.unsolvable_goals[operator.name]
        )
        if unsatisfiable_for_precs:
            logging.debug(
                "Goal unfeasible (negative effects) because all preconditions "
                "cannot be satisfied"
            )
            return FALSE()

        # Get combination of action objects that have already been executed, and with
        # ambiguous types
        operator_params = [p for p in operator.parameters]
        object_conditions = {TRUE()}
        for action in executed_action_ambiguous[str(state)]:
            action_op = action.split("(")[0]
            if operator.name == action_op:
                action_objs = [
                    o.strip()
                    for o in action.split("(")[1][:-1].split(",")
                    if o.strip() != ""
                ]
                action_objs = [problem.object(o) for o in action_objs]

                object_condition = set()
                for param, obj in zip(operator_params, action_objs, strict=True):
                    object_condition.add(Equals(param, obj))
                object_conditions.add(Not(And(*object_condition)))

        E_minus_comb = [
            e_comb
            for e_comb in combinations(goal_eff_minus, length)
            if not np.any(
                [
                    unsolvable.issubset(precs | {str(e) for e in e_comb})
                    for unsolvable in self.unsolvable_goals[operator.name]
                ]
            )
        ]

        for E_minus_tuple in E_minus_comb:
            E_minus = set(E_minus_tuple)
            g = And(
                *(
                    [e for e in E_minus]
                    + list(operator.preconditions)
                    + list(object_conditions)
                )
            )

            if g not in goals:
                goals.add(g)

            if len(goals) > self.max_goals:
                logging.debug(f"Number of generated goals exceeds the maximum "
                              f"limit {self.max_goals}. Some goals are being "
                              f"discarded.")
                break

        if len(goals) == 0:
            processed_goals = [FALSE()]
        else:
            vars = [
                Variable(f"param_{i + 1}", p.type)
                for i, p in enumerate(operator.parameters)
            ]
            subs_map = {p: v for p, v in zip(operator.parameters, vars, strict=True)}

            # TODO: use only necessary inequalities for disambiguating uncertain effects
            inequalities = []
            for v1, v2 in combinations(vars, 2):
                if v1.type.is_compatible(v2.type) or v2.type.is_compatible(v1.type):
                    inequalities.append(Not(Equals(v1, v2)))
            inequalities = And(*inequalities) if inequalities else TRUE()

            processed_goals = [
                Exists(And(g.substitute(subs_map), And(inequalities)), *vars)
                for g in goals
            ]
        return Or(*processed_goals)
