from unified_planning.model import Problem, State

from olam.env import ENV


def construct_problem(domain: Problem, state: State) -> Problem:
    problem = domain.clone()
    assert problem.environment is ENV

    objects_type = dict()
    specialized_types = dict()
    fluent_by_name = {f.name: f for f in problem.fluents}
    for fluent, _ in state.fluents.items():
        original_fluent = fluent_by_name.get(fluent.fluent().name)
        assert len(original_fluent.signature) == len(fluent.args)
        if len(fluent.args) > 0:
            for i, original_obj in enumerate(original_fluent.signature):
                current_obj = fluent.args[i].object()
                if current_obj.name not in objects_type.keys():
                    objects_type[current_obj.name] = original_obj.type
                else:
                    new_type = original_obj.type
                    old_type = objects_type[current_obj.name]
                    if old_type == new_type:
                        continue
                    if old_type.is_compatible(new_type):
                        objects_type[current_obj.name] = new_type
                        specialized_types[current_obj.name] = True

    for obj_name, obj_type in objects_type.items():
        problem.add_object(obj_name, obj_type)

    return problem
