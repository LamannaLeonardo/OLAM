from dataclasses import dataclass
from functools import cached_property

from unified_planning.model import FNode


@dataclass
class SymbolicObservation:
    """
    An observation of a symbolic state, i.e. a set of fluents. For example,
    in classical planning, an observation is a subset of state literals.
    """

    fluents: dict[FNode, FNode]

    def __str__(self):
        return ", ".join(sorted([str(lit) for lit in self.positive_literals]))

    @cached_property
    def positive_literals(self) -> set[FNode]:
        """
        Returns the set of positive literals in the observation.
        :return: set of positive literals
        """
        return {fluent for fluent, value in self.fluents.items() if value.is_true()}

    @cached_property
    def negative_literals(self) -> set[FNode]:
        """
        Returns the set of negative literals in the observation.
        :return: set of negative literals
        """
        return {fluent for fluent, value in self.fluents.items() if value.is_false()}

    def clone(self):
        return SymbolicObservation(self.fluents)
