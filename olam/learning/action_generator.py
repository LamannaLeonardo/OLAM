from dataclasses import dataclass, field
from typing import Any

from unified_planning.model import Object, Problem
from unified_planning.plans import ActionInstance

from olam.modeling.symbolic_observation import SymbolicObservation
from olam.util.executability.executability_cy import (
    compute_all_not_executable_actions_cy,
    compute_not_executable_actions_cy,
)
from olam.util.util import ground_lifted_atoms


@dataclass
class ActionGenerator:
    """
    Adapter class for generating actions in a planning domain.

    Args:
        domain (Problem): The planning problem domain.
        all_grounded_actions (list[str]): A list of all grounded action (string) labels
        all_objects: A dictionary of unified planning problem objects indexed by name
        executed_actions (dict[str, list]): A dictionary where key are state hashes
                                            and values are actions that have been
                                            executed in the state
        executed_actions_ambiguous (dict[Any, Any]): A dictionary where key are tuples
                                                     (object_id, object_type) of objects
                                                     with ambiguous (super)types, and
                                                     values are dictionaries like
                                                     "executed_actions"

    """

    domain: Problem
    all_grounded_actions: list[str]
    all_objects: dict[str, Object]
    executed_actions: Any
    executed_actions_ambiguous: dict[Any, Any]
    _executable_actions: dict[str, list] = None
    _not_executable_actions: list = field(default_factory=list)
    _additional_not_executable: set = field(default_factory=set)
    _action_precs_ratio_cache: dict = field(default_factory=dict)

    def compute_not_executable_actions(
        self,
        state: SymbolicObservation,
        prev_not_executable_actions: list,
        action_list: list[str],
        pre_bot,
        last_failed_action: ActionInstance = None,
    ) -> list[ActionInstance]:
        """
        Compute a list of not executable action (string) labels based on pre_bot

        :parameter state: the current state
        :parameter last_failed_action: the last failed action
        :parameter prev_not_executable_actions: list of not executable action labels
                                                previously computed in the same state.
                                                This is useful to speed up the
                                                computation of the actions between
                                                subsequent failures (where the state
                                                does not change).
        :parameter action_list: list of all grounded actions to be considered
        :parameter pre_bot: disjunction of conjunctions of the form:
                               (p5, p2) or (p1, p4, p6) or (p2) or ...
                               indicating that when (p5 and p2 are false) or (p1 and p4
                               and p6 are false) or p2 is false or ... then the action
                               is not executable

        :return: List of not executable action (string) labels
        """

        pre_bottom_nonempty = {k: v for k, v in pre_bot.items() if len(v) > 0}
        if len(pre_bottom_nonempty) == 0:
            return list()

        current_state = [str(lit) for lit in state.positive_literals]
        action_ids = list(range(len(action_list)))

        # Write action negative preconditions into a json file
        operator_executability_constr = {}

        for action_key, sets in pre_bottom_nonempty.items():
            if len(sets) != 0:
                operator_executability_constr[action_key] = []
                for actual_set in sets:
                    false_precs = [f"not({str(p)})" for p in actual_set]
                    if len(false_precs) > 0:
                        operator_executability_constr[action_key].append(false_precs)

        if last_failed_action is None:
            not_executable_actions = compute_all_not_executable_actions_cy(
                current_state, action_list, action_ids, operator_executability_constr
            )
        else:
            last_failed_action = str(last_failed_action).replace(" ", "")
            last_failed_action_str = (
                f"{last_failed_action} ++ {action_list.index(last_failed_action)}"
            )
            not_executable_actions = compute_not_executable_actions_cy(
                current_state,
                action_list,
                operator_executability_constr,
                last_failed_action_str,
            )

            not_executable_actions |= set(prev_not_executable_actions)

        return not_executable_actions

    def compute_executable_actions(
        self, state: SymbolicObservation
    ) -> tuple[list[str], dict[str, float]]:
        """
        Compute a list of executable action (string) labels in the given state

        :parameter state: the current state

        :return: List of executable action (string) labels
        """

        executable_actions: list[str] = []
        action_precondition_ratio: dict[str, float] = {}

        positive_fluents = {str(fluent) for fluent in state.positive_literals}
        for action in self.all_grounded_actions:
            op_name = action.split("(")[0]
            objs = [
                o.strip()
                for o in action.split("(")[1][:-1].split(",")
                if o.strip() != ""
            ]

            operator = self.domain.action(op_name)

            mapping = {f"param_{i + 1}": o for i, o in enumerate(objs)}
            grounded_precs = list()
            for prec in operator.preconditions:
                new_args = [mapping[str(arg)] for arg in prec.args]

                ground_atom = f"{prec.fluent().name}({', '.join(new_args)})"

                grounded_precs.append(ground_atom)

            # the following line is not working when two lifted precs map to
            # the same grounded prec
            # n_preconditions_satisfied = len(grounded_precs & positive_fluents)
            n_preconditions_satisfied = len(
                [p for p in grounded_precs if p in positive_fluents]
            )

            # % of satisfied precondition
            action_precondition_ratio[action] = n_preconditions_satisfied / len(
                operator.preconditions
            )
            if n_preconditions_satisfied == len(operator.preconditions):
                executable_actions.append(action)

        return sorted(executable_actions), action_precondition_ratio

    def get_learning_actions_precs(
        self,
        state: SymbolicObservation,
        last_failed_action: ActionInstance,
        pre_bot,
        pre_bot_ambiguous,
    ) -> list[str]:
        """
        Return the list of actions for learning preconditions in the current state,
        i.e., the list of "learning actions" (for the preconditions). An action allows
        to learn preconditions if at least an (uncertain) precondition is not satisfied
        in the current state.

        :parameter state: the current state
        :parameter last_failed_action: the last failed action
        :parameter pre_bot: disjunction of conjunctions of the form:
                               (p5, p2) or (p1, p4, p6) or (p2) or ...
                               indicating that when (p5 and p2 are false) or (p1 and p4
                               and p6 are false) or p2 is false or ... then the action
                               is not executable
        :parameter pre_bot_ambiguous: A dictionary where key are tuples
                                      (object_id, object_type) of objects with
                                      ambiguous (super)types, and values are
                                      dictionaries like "pre_bot"

        :return: List of action (string) labels for learning preconditions
        """
        # 1 Compute not executable actions
        self._not_executable_actions = self.compute_not_executable_actions(
            state,
            self._not_executable_actions,
            self.all_grounded_actions,
            pre_bot,
            last_failed_action=last_failed_action,
        )

        # 1.2 Compute additional not executable actions with ambiguous object types
        if last_failed_action is None:
            self._additional_not_executable = set()

        for action_name, P_bot_ambiguous in pre_bot_ambiguous.items():
            if (
                last_failed_action is not None
                and last_failed_action.action.name != action_name
            ):
                continue

            if len(P_bot_ambiguous) == 0:
                continue
            op_ground_actions = [
                a for a in self.all_grounded_actions if a.startswith(f"{action_name}(")
            ]

            for cond, P_bot in P_bot_ambiguous.items():
                args = [None for _ in self.domain.action(action_name).parameters]
                assert len(args) > 0

                for c in cond:
                    param_idx = int(c[0].split("_")[-1]) - 1
                    args[param_idx] = c[1]

                all_pre_bottom = {action_name: P_bot}

                op_ground_actions_cond = [
                    a
                    for a in op_ground_actions
                    if all(
                        x is None or x == y
                        for x, y in zip(
                            args, a.split("(")[1][:-1].split(","), strict=True
                        )
                    )
                ]

                not_executable_of_op = self.compute_not_executable_actions(
                    state,
                    self._not_executable_actions,
                    op_ground_actions_cond,
                    all_pre_bottom,
                )

                self._additional_not_executable |= set(not_executable_of_op)

        not_executable_actions = (
            set(self._not_executable_actions) | self._additional_not_executable
        )

        # 2. Compute executable actions (update the list of executable actions only
        # if the agent changed state, i.e., last_failed_action is None.
        if last_failed_action is None:
            self._executable_actions, self._action_precs_ratio_cache = (
                self.compute_executable_actions(state)
            )

        # 3. Sort remaining learning actions by the ratio of satisfied preconditions,
        # so that the agent firstly tries actions with more satisfied preconditions.
        learning_actions = list(
            set(self.all_grounded_actions)
            - set(not_executable_actions)
            - set(self._executable_actions)
            - set(self.executed_actions_ambiguous[str(state)])
            - set(self.executed_actions[str(state)])
        )
        action_preconditions_ratio = [
            self._action_precs_ratio_cache[a] for a in learning_actions
        ]

        assert (
            len(action_preconditions_ratio) == 0
            or max(action_preconditions_ratio) < 1.0
        )

        return [
            a
            for _, a in sorted(
                zip(action_preconditions_ratio, learning_actions, strict=True),
                reverse=True,
            )
        ]

    def get_learning_actions_effs(
        self,
        state: SymbolicObservation,
        uncertain_positive_effects,
        uncertain_negative_effects,
    ) -> list[str]:
        """
        Return the list of actions for learning effects in the current state,
        i.e., the list of "learning actions" (for the effects). An action allows
        to learn effects if the action preconditions are satisfied and at least an
        (uncertain) effect is not satisfied in the current state.

        :parameter state: the current state
        :parameter uncertain_positive_effects: list of uncertain positive effects
        :parameter uncertain_negative_effects: list of uncertain negative effects

        :return: List of action (string) labels for learning effects
        """

        negative_literals = {str(lit) for lit in state.negative_literals}
        positive_literals = {str(lit) for lit in state.positive_literals}

        learning_actions = []
        action_effects_ratio = []

        for action in self._executable_actions:
            ground_action = self.parse_textual_action(action, self.domain)
            assert ground_action is not None, f"Could not parse action {action}"
            uncertain_pos_grounded = ground_lifted_atoms(
                ground_action, uncertain_positive_effects[ground_action.action.name]
            )
            uncertain_neg_grounded = ground_lifted_atoms(
                ground_action, uncertain_negative_effects[ground_action.action.name]
            )

            pos_ratio = uncertain_pos_grounded & negative_literals
            neg_ratio = uncertain_neg_grounded & positive_literals
            effs_ratio = len(pos_ratio) + len(neg_ratio)
            if effs_ratio > 0:
                if (
                    action
                    not in self.executed_actions[str(state)]
                    | self.executed_actions_ambiguous[str(state)]
                ):
                    learning_actions.append(action)
                    action_effects_ratio.append(effs_ratio)

        return [
            a
            for _, a in sorted(zip(action_effects_ratio, learning_actions, strict=True))
        ]

    def parse_textual_action(
        self, action_instance: str, problem: Problem
    ) -> ActionInstance:
        """
        Get a unified planning problem action instance from a ground action
        (string) label.

        :parameter action_instance: the ground action (string) label
        :parameter problem: the unified planning problem

        :return: Action instance in the unified planning problem
        """
        split = action_instance.split("(")
        operator_name, parameters_raw = split[0], split[1].replace(")", "")
        # operator
        operator = problem.action(operator_name)
        # params
        params = parameters_raw.replace(",", " ").split(" ")
        actual_parameters = [self.all_objects[p] for p in params]
        assert len(actual_parameters) == len(operator.parameters)

        params_retyped = []
        for op_param, actual_param in zip(
            operator.parameters, actual_parameters, strict=True
        ):
            if op_param.type.is_compatible(actual_param.type):
                params_retyped.append(actual_param)
            else:
                params_retyped.append(Object(actual_param.name, op_param.type))

        return ActionInstance(operator, params_retyped)
