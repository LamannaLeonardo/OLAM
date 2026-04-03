import itertools
import logging
from collections import OrderedDict, defaultdict

import numpy as np
from unified_planning.io import PDDLWriter, PDDLReader
from unified_planning.model import Fluent, FNode, Parameter, Problem
from unified_planning.plans import ActionInstance
from unified_planning.shortcuts import BoolType, SequentialSimulator

from olam.learning.action_generator import ActionGenerator
from olam.learning.learner import Learner
from olam.modeling.PDDLenv import PDDLEnv
from olam.modeling.symbolic_observation import SymbolicObservation
from olam.modeling.trajectory import Trajectory
from olam.util.ground_actions_util import get_all_grounded_actions
from olam.util.state_util import construct_problem
from olam.util.util import get_op_relevant_predicates


class OLAM:
    def __init__(self,
                 domain_path: str,
                 planning_timeout: int = 30,
                 max_length: int = 8,
                 max_subproblems: int = 5,
                 max_goals: int = 10000,
                 ):
        """
        Class that implements the Online Learning of Action Models (OLAM) algorithm.

        OLAM learns PDDL action models from interaction with an environment by
        iteratively executing actions, observing state transitions, and refining
        preconditions and effects. The environments are assumed to be fully
        observable and deterministic.

        Args:
            domain_path (str): Input planning domain path, which is required to specify
                                operator and predicates signatures.
            planning_timeout (int): Time limit (in seconds) for each planning process.
            max_length (int): Maximum number of uncertain preconditions and effects to
                                be considered in each goal conjunction
            max_subproblems (int): Maximum number of subproblems to be generated when
                                    handling object types ambiguity, where each
                                    subproblem is associated with a combination of
                                    objects (sub)types.
            max_goals (int): Maximum number of disjunctions in a goal formula used
                              during planning for learning preconditions and effects.
                              When the number of generated goals exceeds this limit,
                              some goals are discarded.

        Example:
            .. code-block:: python

                from amlgym.util.util import empty_domain
                from unified_planning.io import PDDLReader
                from olam.OLAM import OLAM
                from unified_planning.shortcuts import SequentialSimulator
                import unified_planning

                # Disable printing of planning engine credits
                unified_planning.shortcuts.get_environment().credits_stream = None

                domain_ref_path = "olam/benchmarks/domains/blocksworld.pddl"
                problem_path = "olam/benchmarks/problems/blocksworld/1_p00_blocksworld_gen.pddl"
                empty_domain_path = empty_domain(domain_ref_path)
                olam = OLAM(empty_domain_path)

                sim_problem = PDDLReader().parse_problem(domain_ref_path,
                                                         problem_path)
                simulator = SequentialSimulator(sim_problem)
                learned_domain_str, trajectory = olam.run(simulator, max_steps=100)

                print(f"Generated a trajectory with {len(trajectory.observations)} states")
                print(f"Domain learned: {learned_domain_str}")

        """

        domain = PDDLReader().parse_problem(domain_path)
        self.domain = self.normalize_domain(domain)

        # Initialize preconditions and effects
        for operator in self.domain.actions:
            # TODO: manage input knowledge about certain/uncertain preconditions
            assert len(operator.preconditions) == 0, NotImplementedError

            # TODO: manage input knowledge about certain/uncertain effects
            assert len(operator.effects) == 0, NotImplementedError

            # Add all possible precondition of each action
            for precondition in get_op_relevant_predicates(operator, self.domain):
                assert isinstance(precondition, FNode)
                operator.add_precondition(precondition)

        # self.learner = Learner(self.domain)
        self.pre_bot = {operator.name: set() for operator in self.domain.actions}

        # Stores the elements of pre_bottom when the object types are ambiguous, i.e.,
        # the action can fail also due to the parameter types being not valid. This is
        # useful when an object (super)type is ambiguous, which lead to trying
        # attempting action that may act only on some subtypes, therefore the failure
        # may be due just to the input object type (and the actual preconditions may be
        # satisfied despite the action failing).
        self.pre_bot_ambiguous_types = {
            operator.name: dict() for operator in self.domain.actions
        }

        self.uncertain_positive_effects = {
            operator.name: set(get_op_relevant_predicates(operator, self.domain))
            for operator in self.domain.actions
        }
        self.uncertain_negative_effects = {
            operator.name: set(get_op_relevant_predicates(operator, self.domain))
            for operator in self.domain.actions
        }
        self.learned_operators = set()
        self.invalid_actions = set()

        self.PLANNER_CFG = {
            "name": "fast-downward",
            "params": dict(
                fast_downward_translate_options=[
                    "--invariant-generation-max-time",
                    f"{planning_timeout}",
                ],
                fast_downward_search_config="let(hff,ff(),let(hcea,cea(),"
                "lazy_greedy([hff,hcea],"
                "preferred=[hff,hcea])))",
                fast_downward_search_time_limit=f"{planning_timeout}s",
            ),
        }

        self.max_length = max_length

        self.max_subproblems = max_subproblems

        self.max_goals = max_goals

        # Episode-specific information
        self.simulator = None
        self.initial_state = None
        self.problem = None
        self.all_grounded_actions = None
        self.all_objects = None
        self.last_failed_action = None
        self.executed_actions = None
        self.executed_actions_ambiguous = None
        self.learner = None
        self.action_generator = None

    def normalize_domain(self, domain: Problem) -> Problem:
        """
        Remove numeric fluents and rename operator parameters
        """
        normalized_domain = domain.clone()

        # Remove numerical fluent
        normalized_domain.clear_fluents()
        for f in [f for f in domain.fluents if f.type.is_bool_type()]:
            normalized_domain.add_fluent(f)

        # Rename parameters
        for action in normalized_domain.actions:
            new_parameters = OrderedDict()
            for i, p in enumerate(action.parameters):
                new_name = f"param_{i + 1}"
                new_p = Parameter(new_name, p.type)
                new_parameters[new_name] = new_p
            action._parameters = new_parameters
        return normalized_domain

    def reset(self, simulator):

        # Set learning environment
        self.simulator = simulator

        # Get initial state and initialize trajectory
        self.initial_state, _ = self.simulator.reset()

        # Reconstruct the problem based on the domain and the initial state
        self.problem = construct_problem(self.domain, self.initial_state)

        # Data structures
        self.all_grounded_actions = get_all_grounded_actions(self.problem)
        self.all_objects = self.problem.all_objects
        self.last_failed_action = None
        self.executed_actions = defaultdict(set)

        # Store the actions executed in some state when the object types are ambiguous,
        # i.e., the action can fail also due to the parameter types being not valid.
        # This is useful when an object (super)type is ambiguous, which lead to trying
        # attempting action that may act only on some subtypes, therefore the failure
        # may be due just to the input object type (and the actual preconditions may be
        # satisfied despite the action failing). For example, in elevators, if the type
        # of the object slow0-0 is "elevator", then the action
        # "move-up-FAST(slow0-0, ..., ...)" always fail because the actual type of
        # slow0-0 is not valid, however, the actual precondition of move-up-fast may be
        # satisfied, with the action failing just because of the invalid object type.
        self.executed_actions_ambiguous = defaultdict(set)

        self.learned_operators = set()
        self.pre_bot_ambiguous_types = {
            operator.name: defaultdict(set) for operator in self.domain.actions
        }

        self.learner = Learner(self.domain,
                               self.all_objects,
                               max_length=self.max_length,
                               max_subproblems=self.max_subproblems,
                               max_goals=self.max_goals)
        self.action_generator = ActionGenerator(
            self.domain,
            self.all_grounded_actions,
            {obj.name: obj for obj in self.all_objects},
            self.executed_actions,
            self.executed_actions_ambiguous,
        )

        self.invalid_actions = set()

    def run(self, simulator: SequentialSimulator, max_steps: int = 10000):

        simulator = PDDLEnv(simulator)
        self.reset(simulator)

        state = self.infer_state_types(self.initial_state.clone())
        traj = Trajectory(observations=[state], actions=list())

        plan = None
        for _ in range(max_steps):
            # Action for learning preconditions
            executing_learning_action = False
            actions_for_preconditions = (
                self.action_generator.get_learning_actions_precs(
                    state,
                    self.last_failed_action,
                    self.pre_bot,
                    self.pre_bot_ambiguous_types,
                )
            )
            actions_for_effects = self.action_generator.get_learning_actions_effs(
                state, self.uncertain_positive_effects, self.uncertain_negative_effects
            )
            learning_actions = actions_for_preconditions + actions_for_effects

            if len(learning_actions) > 0:
                learning_action_label = learning_actions.pop(0)
                executing_learning_action = True

                learning_action = self.action_generator.parse_textual_action(
                    learning_action_label, self.action_generator.domain
                )
                obj_types = [
                    self.problem.object(str(o)).type
                    for o in learning_action.actual_parameters
                ]
                action_types = [p.type for p in learning_action.action.parameters]
                compatible_types = [
                    p_type.is_compatible(o_type)
                    for o_type, p_type in zip(obj_types, action_types, strict=True)
                ]

                if not np.all(compatible_types):
                    self.executed_actions_ambiguous[str(state)].add(
                        learning_action_label
                    )

                else:
                    self.executed_actions[str(state)].add(learning_action_label)

                op = self.action_generator.parse_textual_action(
                    learning_action_label, self.action_generator.domain
                )
                logging.debug("Executing learning action")
            else:
                if plan is None or len(plan.actions) == 0:
                    # order the operators based on the number of the preconditions
                    operators_to_refine = [
                        a.name
                        for a in sorted(
                            self.domain.actions, key=lambda x: len(x.preconditions)
                        )
                        if a.name not in self.learned_operators
                    ]
                    for operator_name in operators_to_refine:
                        plan = self.learner.plan_for_operator(
                            state,
                            self.domain.action(operator_name),
                            self.uncertain_positive_effects,
                            self.uncertain_negative_effects,
                            self.pre_bot,
                            self.pre_bot_ambiguous_types,
                            self.executed_actions_ambiguous,
                            self.problem,
                            self.invalid_actions,
                            self.PLANNER_CFG,
                        )

                        if plan is None:
                            logging.debug(f"Operator {operator_name} learned!")
                            self.learned_operators.add(operator_name)
                        else:
                            assert len(plan.actions) > 0
                            break
                if plan is None:
                    break
                else:
                    assert len(plan.actions) > 0
                    op = plan.actions[0]
                    logging.debug("Executing planned action")
                    plan.actions.remove(op)
                    if len(plan.actions) == 0:
                        plan = None

            # Construct the operator according to the problem object for compatiblity
            # with the simulator
            operator_name, actual_params = op.action.name, op.actual_parameters
            next_state, reward, done, truncated, info = self.simulator.step(op)
            next_state = self.infer_state_types(next_state)

            executable = next_state is not None
            operator = self.domain.action(operator_name)
            action_instance = ActionInstance(operator, actual_params)
            if executable:
                logging.debug(f"Successfully executed action {action_instance}")

                # Update trajectory
                traj.add_action(action_instance)
                traj.add_obs(next_state)

                # Learn effects from trajectory
                updated_effs = self.learn_effects(action_instance, state, next_state)

                # Learn preconditions from trajectory
                updated_precs = self.learn_preconditions(action_instance, state)

                updated_model = updated_precs or updated_effs
                if updated_model:
                    self.learner.goal_gen.unsolvable_goals = defaultdict(set)
                    self.learned_operators = set()
                # Update the state
                state = next_state

                self.last_failed_action = None

                # if a learning action is interleaved while executing a plan, and the
                # learning action succeeds, then the agent must replan
                if executing_learning_action:
                    plan = None
            else:
                # Failed action
                logging.debug(f"Failed to execute action {action_instance}")

                # If the action preconditions are satisfied, then the action must
                # involve some wrong object type. This can happen when an object
                # (sub)type is not inferred which cause the agent trying ground actions
                # with all possible subtypes, possibly attempting to execute actions
                # with invalid object types.
                params_map = {
                    p: o
                    for p, o in zip(
                        action_instance.action.parameters,
                        action_instance.actual_parameters,
                        strict=True,
                    )
                }
                ground_precs = {
                    str(p.substitute(params_map))
                    for p in action_instance.action.preconditions
                }
                pos_literals = {str(lit) for lit in state.positive_literals}
                if ground_precs.issubset(pos_literals):
                    obj_types = [
                        self.problem.object(str(o)).type
                        for o in action_instance.actual_parameters
                    ]
                    action_types = [p.type for p in action_instance.action.parameters]
                    compatible_types = [
                        p_type.is_compatible(o_type)
                        for o_type, p_type in zip(obj_types, action_types, strict=True)
                    ]
                    assert not all(compatible_types), (
                        "An action with all satisfied preconditions has failed, "
                        "and the action object types are *not* ambiguous."
                    )

                    logging.debug(
                        f"Adding {action_instance} to the list of invalid actions."
                    )

                    # TODO: open issue in unified planning, the if cond should
                    #  not be necessary
                    if str(action_instance) not in {
                        str(a) for a in self.invalid_actions
                    }:
                        self.invalid_actions.add(action_instance)

                else:
                    self.learn_preconditions_from_failed_action(action_instance, state)

                self.last_failed_action = action_instance

        # TODO: open issue in amlgym as the syntactic metrics raise an error
        #  when no effects section is specified for an operator
        domain = self.domain.clone()
        dummy = Fluent("dummy", BoolType())
        for a in domain.actions:
            a.add_effect(dummy, True)
        domain_str = PDDLWriter(domain).get_domain()
        domain_str = domain_str.replace("(dummy)", "")

        return domain_str, traj

    def infer_state_types(
        self, state: SymbolicObservation
    ) -> SymbolicObservation | None:
        if state is None:
            return None
        fluents_typed = dict()
        for fluent, value in state.fluents.items():
            fluent_typed = self.problem.fluent(fluent.fluent().name)
            objects = []
            for obj in fluent.args:
                objects.append(self.problem.object(str(obj)))

            fluents_typed[fluent_typed(*objects)] = value
        return SymbolicObservation(fluents_typed)

    def learn_effects(
        self,
        action: ActionInstance,
        state: SymbolicObservation,
        new_state: SymbolicObservation,
    ) -> bool:
        """
        Learn the effects of each action from the given trajectory
        """
        delta_pos = state.negative_literals & new_state.positive_literals
        delta_neg = state.positive_literals & new_state.negative_literals

        operator = action.action

        updated = False

        # Fluents that become true are certain positive effects of the operator
        for a in delta_pos:
            fluents_add = self.lift_ground_atoms(action, [str(a)])
            if len(fluents_add) > 1:
                logging.debug(f"Ambiguous positive effect: {fluents_add}")
                if (
                    len(
                        set(fluents_add).intersection(
                            set(
                                [
                                    str(f.fluent)
                                    for f in operator.effects
                                    if f.value.is_true()
                                ]
                            )
                        )
                    )
                    > 0
                ):
                    continue
                fluents_add = [
                    f
                    for f in fluents_add
                    if f
                    in {str(p) for p in self.uncertain_positive_effects[operator.name]}
                ]
                if len(fluents_add) > 1:
                    continue
            for fluent in fluents_add:
                if fluent in {
                    str(f) for f in self.uncertain_positive_effects[operator.name]
                }:
                    # add the fluent as a positive effect and remove it from the
                    # uncertain positive effects
                    up_fluent = next(
                        p
                        for p in self.uncertain_positive_effects[operator.name]
                        if str(p) == fluent
                    )
                    operator.add_effect(fluent=up_fluent, value=True)
                    updated = True
                    logging.info(
                        f"Operator {operator.name} adding positive effect {str(fluent)}"
                    )
                    self.uncertain_positive_effects[operator.name].remove(up_fluent)
                    logging.debug(
                        f"Operator {operator.name} removing uncertain positive "
                        f"effect {str(fluent)}"
                    )

                    # remove this fluent from the uncertain negative effects
                    if fluent in {
                        str(p) for p in self.uncertain_negative_effects[operator.name]
                    }:
                        self.uncertain_negative_effects[operator.name].remove(up_fluent)
                        logging.debug(
                            f"Operator {operator.name} removing uncertain negative "
                            f"effect {str(fluent)}"
                        )

        # The fluents that become false are certain negative effects of the operator
        for d in delta_neg:
            fluents_del = self.lift_ground_atoms(action, [str(d)])
            if len(fluents_del) > 1:
                logging.debug(f"Ambiguous atoms - [del_] - {fluents_del}")
                if (
                    len(
                        set(fluents_del).intersection(
                            set(
                                [
                                    str(f.fluent)
                                    for f in operator.effects
                                    if f.value.is_false()
                                ]
                            )
                        )
                    )
                    > 0
                ):
                    continue
                fluents_del = [
                    f
                    for f in fluents_del
                    if f
                    in {str(p) for p in self.uncertain_negative_effects[operator.name]}
                ]
                if len(fluents_del) > 1:
                    continue
            for fluent in fluents_del:
                if fluent in {
                    str(p) for p in self.uncertain_negative_effects[operator.name]
                }:
                    # add the fluent as a negative effect and remove it from the
                    # uncertain negative effects
                    up_fluent = next(
                        p
                        for p in self.uncertain_negative_effects[operator.name]
                        if str(p) == fluent
                    )
                    operator.add_effect(fluent=up_fluent, value=False)
                    updated = True

                    logging.info(
                        f"Operator {operator.name} adding negative effect {str(fluent)}"
                    )
                    self.uncertain_negative_effects[operator.name].remove(up_fluent)
                    logging.debug(
                        f"Operator {operator.name} removing uncertain negative "
                        f"effect {str(fluent)}"
                    )

                    # remove this fluent from the uncertain positive effects
                    if fluent in {
                        str(p) for p in self.uncertain_positive_effects[operator.name]
                    }:
                        self.uncertain_positive_effects[operator.name].remove(up_fluent)
                        logging.debug(
                            f"Operator {operator.name} removing uncertain positive "
                            f"effect {str(fluent)}"
                        )

        # remove positive literals from the uncertain negative effects
        for p in new_state.positive_literals:
            fluents_pos = self.lift_ground_atoms(action, [str(p)])
            if len(fluents_pos) > 1:
                continue

            for fluent in fluents_pos:
                if fluent in {
                    str(p) for p in self.uncertain_negative_effects[operator.name]
                }:
                    up_fluent = next(
                        p
                        for p in self.uncertain_negative_effects[operator.name]
                        if str(p) == fluent
                    )
                    self.uncertain_negative_effects[operator.name].remove(up_fluent)
                    logging.info(
                        f"Operator {operator.name} removing uncertain negative "
                        f"effect {str(fluent)}"
                    )

        # remove negative literals from the uncertain positive effects
        for n in new_state.negative_literals:
            fluents_neg = self.lift_ground_atoms(action, [str(n)])
            if len(fluents_neg) > 1:
                continue

            for fluent in fluents_neg:
                if fluent in {
                    str(p) for p in self.uncertain_positive_effects[operator.name]
                }:
                    up_fluent = next(
                        p
                        for p in self.uncertain_positive_effects[operator.name]
                        if str(p) == fluent
                    )

                    self.uncertain_positive_effects[operator.name].remove(up_fluent)
                    logging.info(
                        f"Operator {operator.name} removing uncertain positive "
                        f"effect {str(fluent)}"
                    )

        return updated

    def learn_preconditions(
        self, action: ActionInstance, obs: SymbolicObservation
    ) -> bool:
        """
        Learn the preconditions of each action from the given trajectory
        """
        # logging.debug(f"Learning preconditions for action {action}...")
        operator = action.action
        updated = False

        # fluents = self.lift_ground_atoms(action, obs.negative_literals)
        fluents = self.lift_ground_atoms(
            action, [str(lit) for lit in obs.negative_literals]
        )
        precs = {str(p) for p in operator.preconditions}
        for fluent in sorted(fluents):
            if fluent in precs:
                up_fluent = next(p for p in operator.preconditions if str(p) == fluent)

                operator.preconditions.remove(up_fluent)
                updated = True
                assert len(operator.preconditions) > 0, "No remaining preconditions"
                logging.info(
                    f"Operator {operator.name} removing uncertain "
                    f"precondition {str(fluent)}"
                )

            # Update pre bottom:
            # - remove non-preconditions from each element of pre_bot
            self.pre_bot[operator.name] = {
                frozenset(set(P) - {fluent}) for P in self.pre_bot[operator.name]
            }

            # - filter out empty sets
            self.pre_bot[operator.name] = {
                s for s in self.pre_bot[operator.name] if len(s) > 0
            }

            # - remove non-minimal sets from pre_bot
            self.pre_bot[operator.name] = {
                s
                for s in self.pre_bot[operator.name]
                if not any(t < s for t in self.pre_bot[operator.name])
            }

            # Update pre bottom with ambiguous object types:
            pre_bottom_ambiguous = self.pre_bot_ambiguous_types[operator.name]
            for obj_bind in pre_bottom_ambiguous:
                # - remove non-preconditions
                pre_bottom_ambiguous[obj_bind] = {
                    frozenset(set(P) - {fluent}) for P in pre_bottom_ambiguous[obj_bind]
                }
                # - filter out empty sets
                pre_bottom_ambiguous[obj_bind] = {
                    s for s in pre_bottom_ambiguous[obj_bind] if len(s) > 0
                }
                # - remove non-minimal sets
                pre_bottom_ambiguous[obj_bind] = {
                    s
                    for s in pre_bottom_ambiguous[obj_bind]
                    if not any(t < s for t in pre_bottom_ambiguous[obj_bind])
                }

        return updated

    def learn_preconditions_from_failed_action(
        self, action: ActionInstance, state: SymbolicObservation
    ) -> None:
        """
        Learn the preconditions of each action from the given trajectory
        """

        # Get lifted atoms corresponding to unsatisfied preconditions, i.e.,
        # action preconditions that do not hold in the state where the action
        # execution has failed
        params_map = {
            p: o
            for p, o in zip(
                action.action.parameters, action.actual_parameters, strict=True
            )
        }
        ground_precs = {
            str(p.substitute(params_map)) for p in action.action.preconditions
        }
        pos_literals = {str(lit) for lit in state.positive_literals}
        unsat_preconds = ground_precs - pos_literals
        unsat_preconds = self.lift_ground_atoms(
            action, list(unsat_preconds)
        ).intersection({str(p) for p in action.action.preconditions})
        assert len(unsat_preconds) > 0, "No unsatisfied preconditions"

        # Check there are no ambiguous object types, since the actual preconditions
        # might be satisfied, but the action be not executable just because of the
        # input object type is not correct due to its ambiguity.
        # For example, in domain barman, if the object shot2 has ambiguous supertype
        # "container" (with subtypes 'shaker' and 'shot'), then
        # clean-shaker(right, left, shot2) always fails, even if all preconditions
        # (but the object type) are satisfied.
        obj_types = [self.problem.object(str(o)).type for o in action.actual_parameters]
        action_types = [p.type for p in action.action.parameters]
        compatible_types = [
            p_type.is_compatible(o_type)
            for o_type, p_type in zip(obj_types, action_types, strict=True)
        ]
        if not np.all(compatible_types):
            ambiguous_objs = {
                f"param_{i + 1}": str(o)
                for i, o in enumerate(action.actual_parameters)
                if not compatible_types[i]
            }
            ambiguous_objs = frozenset(ambiguous_objs.items())

            self.pre_bot_ambiguous_types[action.action.name][ambiguous_objs].add(
                frozenset(unsat_preconds)
            )

            # remove any of the sets that are subsets of the new preconditions to add
            sets_to_remove = set()
            for element in self.pre_bot_ambiguous_types[action.action.name][
                ambiguous_objs
            ]:
                if unsat_preconds.issubset(element) and element != unsat_preconds:
                    sets_to_remove.add(element)

            self.pre_bot_ambiguous_types[action.action.name][
                ambiguous_objs
            ].difference_update(sets_to_remove)

            updated_set = self.pre_bot_ambiguous_types[action.action.name][
                ambiguous_objs
            ]
            assert len(
                {s for s in updated_set if not any(t < s for t in updated_set)}
            ) == len(updated_set)

        else:
            # Add the false fluents to the set of possible preconditions
            self.pre_bot[action.action.name].add(frozenset(unsat_preconds))
            sets_to_remove = set()

            # remove any of the sets that are subsets of the new preconditions to add
            for element in self.pre_bot[action.action.name]:
                if unsat_preconds.issubset(element) and element != unsat_preconds:
                    sets_to_remove.add(element)

            self.pre_bot[action.action.name].difference_update(sets_to_remove)

    def lift_ground_atoms(
        self, ground_action: ActionInstance, ground_atoms: list[str]
    ) -> set[str]:
        """
        Lift the atoms of the grounded fluents
        """

        lifted_atoms = set()

        action_args = {str(arg) for arg in ground_action.actual_parameters}

        for g in ground_atoms:
            atom = str(g).split("(")[0]

            objs = []
            if "(" in g:
                objs = [
                    o.strip()
                    for o in str(g).split("(")[1][:-1].split(",")
                    if o.strip() != ""
                ]

            if not set(objs).issubset(action_args):
                continue

            operator = self.domain.action(ground_action.action.name)
            objs_to_params = [
                [
                    p.name
                    for o2, p in zip(
                        ground_action.actual_parameters,
                        operator.parameters,
                        strict=True,
                    )
                    if str(o) == str(o2)
                ]
                for o in objs
            ]

            for comb in itertools.product(*objs_to_params):
                if len(comb) > 0:
                    lifted_atoms.add(f"{atom}({', '.join(comb)})")
                else:
                    lifted_atoms.add(f"{atom}")

        return lifted_atoms
