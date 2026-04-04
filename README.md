# Online Learning of Action Models for PDDL Planning

<div style="display: flex; gap: 10px;">

  <a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" height="20"/></a>

  <a href="https://pypi.python.org/pypi/olam" target="_blank">
    <img src="https://badge.fury.io/py/olam.svg" height="20"/></a>

</div>


Refactored implementation of the **Online Learning of Action Models (OLAM)** algorithm, presented at **IJCAI 2021** ([paper](https://www.ijcai.org/proceedings/2021/0566.pdf)).

OLAM learns PDDL action models online — interleaving planning and execution — without requiring pre-collected trajectories. Starting from a domain skeleton (predicates and operator signatures with empty preconditions and effects), it incrementally refines the action model as the agent acts in a simulated environment.

> The original code used for the paper experiments is preserved at the [`ijcai-2021`](https://github.com/LamannaLeonardo/OLAM/tree/ijcai-2021) branch.


## Installation

### From PyPI

```bash
pip install olam
```

### From [AMLGym](https://amlgym.readthedocs.io/en/latest/index.html)
```bash
pip install amlgym
```

Please refer to [OLAM integration in AMLGym](https://amlgym.readthedocs.io/en/latest/active_algorithms/olam.html)
for a usage example.


### For developers

Clone the repository and install in editable mode:

```bash
git clone https://github.com/LamannaLeonardo/OLAM.git
cd OLAM
pip install -e .
```


---

## Quick Start

```python
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import SequentialSimulator
from amlgym.benchmarks import get_domain_path, get_problems_path
from amlgym.util.util import empty_domain

from olam.OLAM import OLAM

# Instantiate a simulated environment from a PDDL domain and problem
domain = 'blocksworld'
domain_ref_path = get_domain_path(domain)
problem_path = get_problems_path(domain, kind='learning')[0]
problem = PDDLReader().parse_problem(domain_ref_path, problem_path)
env = SequentialSimulator(problem=problem)

# Get an input domain path with predicates and operators signature
input_domain_path = empty_domain(domain_ref_path)

# Run the OLAM algorithm
olam = OLAM(input_domain_path)
domain_learned, trajectory = olam.run(env, max_steps=100)

# Print learned domain and produced trajectory
print("##################### Learned domain #####################")
print(domain_learned)

print("################# Generated trajectory ##################")
print(trajectory)
```

## Benchmark Results
OLAM is evaluated on 23 classical planning domains drawn from the IPC benchmark suite. Each domain is paired with a set of 10 problem instances used for incremental online learning.
The following table reports for each domain the *syntactic precision and recall* of the learned model and CPU time (seconds) averaged over all instances.

| Domain | Precision | Recall | CPU Time (s) |
|---|---|---|---|
| Barman | 0.98 | 1.00 | 385.92 |
| Blocksworld | 1.00 | 1.00 | 3.57 |
| Depots | 0.98 | 1.00 | 5.52 |
| Driverlog | 0.94 | 1.00 | 7.67 |
| Elevators | 0.89 | 1.00 | 183.89 |
| Ferry | 0.93 | 1.00 | 2.07 |
| Floortile | 0.83 | 1.00 | 5.17 |
| Gold-Miner | 0.81 | 1.00 | 15.76 |
| Grid | 0.82 | 1.00 | 5.06 |
| Gripper | 1.00 | 1.00 | 0.43 |
| Hanoi | 0.88 | 1.00 | 0.39 |
| Matching-BW | 0.99 | 1.00 | 66.49 |
| Miconic | 1.00 | 1.00 | 3.99 |
| N-Puzzle | 0.88 | 1.00 | 0.36 |
| NoMystery | 0.92 | 1.00 | 1.46 |
| Parking | 0.89 | 1.00 | 2.49 |
| Rover | 0.83 | 0.88 | 48.19 |
| Satellite | 1.00 | 1.00 | 3.57 |
| Sokoban | 0.88 | 1.00 | 20.39 |
| Spanner | 0.93 | 1.00 | 5.54 |
| TPP | 0.95 | 1.00 | 366.42 |
| Transport | 0.93 | 1.00 | 5.56 |
| Zenotravel | 1.00 | 1.00 | 6.41 |

---

## Citation

If you find OLAM useful for your research, please cite the following paper:

```bibtex
@inproceedings{ijcai2021-566,
  title     = {Online Learning of Action Models for PDDL Planning},
  author    = {Lamanna, Leonardo and Saetti, Alessandro and Serafini, Luciano
               and Gerevini, Alfonso and Traverso, Paolo},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  publisher = {International Joint Conferences on Artificial Intelligence},
  pages     = {4112--4118},
  year      = {2021},
  doi       = {10.24963/ijcai.2021/566},
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE.md) file for details.

---

## Acknowledgements

Refactored with the help of [Ejdis Gjinika](https://github.com/ejdisgjinika).
