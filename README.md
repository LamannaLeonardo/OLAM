# Online Learning of Action Models for PDDL Planning
<div style="display: flex; gap: 10px;">
   
  <a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" height="20"/></a>
    
  <a href="https://pypi.python.org/pypi/olam" target="_blank">
    <img src="https://badge.fury.io/py/olam.svg" height="20"/></a>
    
[//]: # (  <a href="https://amlgym.readthedocs.io/en/latest/" target="_blank">)

[//]: # (    <img src="https://readthedocs.org/projects/amlgym/badge/?version=latest" height="20"/></a>)

</div>

This repository contains the *refactored* code of the 
Online Learning of Action Models (OLAM) algorithm presented at IJCAI 2021, 
for details about the method please see 
the [paper](https://www.ijcai.org/proceedings/2021/0566.pdf). The previous code used 
for the paper experiments is available at this 
[link](https://github.com/LamannaLeonardo/OLAM/tree/ijcai-2021).


## Example Usage
```python
import unified_planning
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import SequentialSimulator
from amlgym.util.util import empty_domain
from olam.OLAM import OLAM

# Disable printing of planning engine credits
unified_planning.shortcuts.get_environment().credits_stream = None

domain_ref_path = "olam/benchmarks/domains/blocksworld.pddl"
problem_path = "olam/benchmarks/problems/blocksworld/1_p00_blocksworld_gen.pddl"
empty_domain_path = empty_domain(domain_ref_path) # remove preconditions/effects
olam = OLAM(empty_domain_path)

sim_problem = PDDLReader().parse_problem(domain_ref_path,
                                         problem_path)
simulator = SequentialSimulator(sim_problem)
learned_domain_str, trajectory = olam.run(simulator, max_steps=100)

print(f"Generated a trajectory with {len(trajectory.observations)} states")
print(f"Domain learned: {learned_domain_str}")
```

## Installation for developers

Clone this repository and install in developer mode:

```
pip install -e .
```


## Citations
```
@inproceedings{ijcai2021-566,
  title     = {Online Learning of Action Models for PDDL Planning},
  author    = {Lamanna, Leonardo and Saetti, Alessandro and Serafini, Luciano and Gerevini, Alfonso and Traverso, Paolo},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {4112--4118},
  year      = {2021},
  doi       = {10.24963/ijcai.2021/566},
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](/LICENSE.md) file for details.

## Acknowledgements
This code has been refactored with the help of [Ejdis Gjinika](https://github.com/ejdisgjinika)