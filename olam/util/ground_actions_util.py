from itertools import product

from unified_planning.model import (
    Problem,
)


def get_all_grounded_actions(problem: Problem) -> list[str]:

    standard_objects_type = {obj: [obj.type] for obj in problem.all_objects}
    additional_objects_type = {
        obj: problem.user_types_hierarchy[obj.type]
        for obj in problem.all_objects
        if len(problem.user_types_hierarchy[obj.type]) > 0
    }

    if len(additional_objects_type) > 0:
        all_objects_type = {
            k: list(
                set(
                    standard_objects_type.get(k, [])
                    + additional_objects_type.get(k, [])
                )
            )
            for k in set(standard_objects_type) | set(additional_objects_type)
        }
    else:
        all_objects_type = standard_objects_type

    grounding_schema = {action.name: [] for action in problem.actions}
    for action in problem.actions:
        params_type_list = [param.type for param in action.parameters]

        candidate_parameters = {i: set() for i, _ in enumerate(params_type_list)}
        for i, operator_type in enumerate(params_type_list):
            # TODO simplify or + is_compatible
            for obj, obj_types in all_objects_type.items():
                if any(operator_type.is_compatible(t) for t in obj_types):
                    candidate_parameters[i].add(obj.name)

        all_tuples_strings = [
            f"{','.join(t)}"
            for t in product(
                *[candidate_parameters[i] for i in sorted(candidate_parameters.keys())]
            )
        ]
        assert len(all_tuples_strings) > 0, f"No grounding for action: {action.name}"
        grounding_schema[action.name] = all_tuples_strings

    grounded_actions = [
        f"{name.lower()}({objs})"
        for name, params in grounding_schema.items()
        for objs in params
    ]

    return grounded_actions
