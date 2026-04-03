import os
from pathlib import Path

import pandas as pd
import pytest
from amlgym.metrics import syntactic_precision, syntactic_recall
from olam.util.util import empty_domain
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import SequentialSimulator

from olam.modeling.PDDLenv import PDDLEnv
from olam.OLAM import OLAM
from olam.util.util import del_numeric_fluents


def run_olam(domain_ref_path: str, problems_path: str):

    empty_domain_path = empty_domain(domain_ref_path)
    olam = OLAM(empty_domain_path)
    os.remove(empty_domain_path)

    learned_domain_str = None
    for problem in sorted(
        os.listdir(problems_path), key=lambda x: int(x.split("_")[0])
    ):
        env_problem = PDDLReader().parse_problem(domain_ref_path,
                                                 os.path.join(problems_path, problem))
        env_problem = del_numeric_fluents(env_problem)

        simulator = SequentialSimulator(env_problem)
        learned_domain_str, _ = olam.run(simulator)

    tmp_path = "tmp.pddl"
    with open(tmp_path, "w") as f:
        f.write(learned_domain_str)
    metrics = {
        "precision": syntactic_precision(tmp_path, domain_ref_path),
        "recall": syntactic_recall(tmp_path, domain_ref_path),
    }
    os.remove(tmp_path)

    return metrics


@pytest.mark.parametrize(
    "domain", pd.read_excel(Path(__file__).parent / "benchmark.xlsx")["Domain"]
)
def test_polam_learning(domain, benchmarks):

    domain_ref_path = f"{Path(__file__).parent}/domains/{domain}.pddl"
    problems_path = f"{Path(__file__).parent}/problems/{domain}"

    metrics = run_olam(domain_ref_path, problems_path)
    target = benchmarks[domain]

    assert metrics["precision"]["mean"] == pytest.approx(target["precision"])
    assert metrics["recall"]["mean"] == pytest.approx(target["recall"])
