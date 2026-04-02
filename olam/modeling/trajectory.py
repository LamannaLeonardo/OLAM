from dataclasses import dataclass

from unified_planning.plans import ActionInstance

from olam.modeling.symbolic_observation import SymbolicObservation


@dataclass
class Trajectory:
    observations: list[SymbolicObservation]
    actions: list[
        ActionInstance
    ]  # TODO: add action supertype to allow different action formats

    def add_obs(self, obs: SymbolicObservation) -> None:
        """
        Add an observation to the trajectory
        :param obs: observation to be added
        :return:
        """
        self.observations.append(obs)

    def add_action(self, a: ActionInstance) -> None:
        """
        Add an action to the trajectory
        :param a: action to be added
        :return:
        """
        self.actions.append(a)

    def __str__(self):
        trajectory_str = ""
        for i, obs in enumerate(self.observations):
            trajectory_str += "(:state "
            for index, literal in enumerate(obs.fluents):
                if index == len(obs.fluents) - 1:
                    trajectory_str += f"{literal})"
                else:
                    trajectory_str += f"{literal}, "
            if i < len(self.observations) - 1:
                trajectory_str += f"\n (action {self.actions[i].action.name} )\n"
        return trajectory_str

    def __repr__(self):
        return str(self)

    def write(self, file_path):
        with open(file_path, "w") as f:
            f.write(str(self))
