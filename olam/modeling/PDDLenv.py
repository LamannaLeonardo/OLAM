import abc
import random
from dataclasses import dataclass
from typing import Any, TypeVar

import gymnasium as gym
from typing_extensions import SupportsFloat
from unified_planning.model import Action, Fluent, FNode, Object, Problem, State
from unified_planning.plans import ActionInstance
from unified_planning.shortcuts import BoolType, SequentialSimulator, UserType

from olam.modeling.symbolic_observation import SymbolicObservation

ObsType = TypeVar("ObsType")
UnknownType = UserType("Unknown")


@dataclass
class PDDLEnv(gym.Env, abc.ABC):
    """
    A Gymnasium environment which simulates a PDDL problem
    through the unified-planning sequential simulator.
    """

    # The PDDL problem in unified-planning
    _problem: Problem

    # The environment state
    _state: State

    # The environment state fluents
    _fluents: list[FNode]

    # The environment simulation engine
    _simulator: SequentialSimulator

    # Environment seed for reproducibility
    seed: int | None = None

    def __init__(
        self, simulator: SequentialSimulator, observability: float = 1.0
    ) -> None:
        """
        Set environment state and seed through :meth: reset for reproducibility
        """

        # Set random seed for reproducibility
        if self.seed is not None:
            super().reset(seed=self.seed)

        # Set the environment simulation engine
        self._simulator = simulator

        # Instantiate the environment state from the simulator
        self._state = self._simulator.get_initial_state()

        # Get the list of state fluents
        self._problem = simulator._problem
        self._fluents = list(self._problem.initial_values.keys())

        # Observability ratio
        self.observability = observability

        self._unknown_type = UserType("Unknown")

        # Define fluent and objects with unknown types. They are used when returning
        # an observation of the environment state without providing information
        # about the object types.
        self._unknown_objs = {
            o.name: Object(o.name, self._unknown_type)
            for o in self._problem.all_objects
        }
        self._unknown_atoms = {
            f.fluent().name: Fluent(
                f.fluent().name,
                BoolType(),
                **{f"p{i}": self._unknown_type for i in range(len(f.args))},
            )
            for f in self._fluents
        }
        self._unknown_fluents = dict()
        for f in self._fluents:
            atom = self._unknown_atoms[f.fluent().name]
            objs = [self._unknown_objs[str(o)] for o in f.args]
            self._unknown_fluents[str(f)] = atom(*objs)

    def step(
        self, action: ActionInstance | str
    ) -> tuple[ObsType | None, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Executes an action in the environment and returns the observation
        and reward associated with the next environment state.
        """
        a = self.parse_action_str(action)

        # The action may not be valid or applicable in a certain state
        if a is None or not self._simulator._is_applicable(self._state, a, None):
            return None, 0.0, False, False, self._get_info()

        next_state = self._simulator.apply(self._state, a)

        # Get the reward r(s,a,s') after executing action `a`
        # in state `s` and reaching `s'`
        reward = self._reward_fn(self._state, a, next_state)

        # Update current state (required for returning the state observation)
        self._state = next_state

        # There is no goal currently. TODO: add goal?
        done = truncated = False

        return self._get_obs(), reward, done, truncated, self._get_info()

    def _get_obs(self) -> SymbolicObservation:
        """
        Return an observation of the current environment state, which consists
        of a subset of state fluents
        Note observations are returned after executing :meth: step and :meth: reset.
        :return: observation of the current environment state
        """

        obs_fluents = {
            self._unknown_fluents[str(f)]: self._state.get_value(f)
            for f in self._fluents
            if random.random() < self.observability
        }

        return SymbolicObservation(obs_fluents)

    def _get_info(self) -> dict[str, Any]:
        """
        Return an additional information dictionary after
        executing :meth: step and :meth: reset.
        :return: additional information about the environment
        """
        return dict()

    # TODO: add reward function?
    def _reward_fn(self, s: State, a: Action, sp: State) -> float:
        """
        Reward function :math: R: S \times A \times S \rightarrow \mathbb{R}^+.
        Note this method should take into account the value of self.goal_states and work
        with possibly different sets of goal states.
        :param s: previous state
        :param a: executed action
        :param sp: current state
        :return: reward for executing action 'a' in state 's' and reaching state 'sp'
        """
        return 0.0

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Resets the environment initial and goal states, and returns the observation of
        the initial state and the (possibly empty) additional information dictionary.
        """
        super().reset(seed=seed)

        # something = options.get('something', None)

        return self._get_obs(), self._get_info()  # Observation, info dict

    # TODO: add rendering?
    def render(self):
        if self.render_mode == "human":
            raise NotImplementedError
        elif self.render_mode == "rgb_array":
            raise NotImplementedError

    def close(self) -> None:
        """
        Cleanup when the environment is closed.
        """
        # print("Environment closed.")
        pass

    def parse_action_str(self, action: ActionInstance | str) -> ActionInstance | None:
        action_instance_str = str(action)
        split = action_instance_str.split("(")
        operator_name, parameters_raw = split[0], split[1].strip()[:-1]
        # operator
        operator = self._problem.action(operator_name)
        # params
        param_names = [p.strip() for p in parameters_raw.split(",") if p.strip() != ""]
        params = [self._problem.object(p) for p in param_names]

        if not all(
            op_param.type.is_compatible(param.type)
            for op_param, param in zip(operator.parameters, params, strict=True)
        ):
            return None

        return ActionInstance(operator, params)
