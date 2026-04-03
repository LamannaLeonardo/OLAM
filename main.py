import json
import logging
import os
import pprint
import random
import sys
import time
from pathlib import Path

import numpy as np
import unified_planning
from amlgym.metrics import syntactic_precision, syntactic_recall
from olam.util.util import empty_domain
from unified_planning.io import PDDLReader, PDDLWriter
from unified_planning.shortcuts import SequentialSimulator

from olam.OLAM import OLAM
from olam.util.util import del_numeric_fluents

# Disable printing of planning engine credits to avoid overloading stdout
unified_planning.shortcuts.get_environment().credits_stream = None


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

if __name__ == "__main__":
    # Set random seed
    RANDOM_SEED = 0
    BENCHMARK_DIR = "olam/benchmarks"
    RES_DIR = "res"
    os.makedirs(RES_DIR, exist_ok=True)

    domains = [d.split(".")[0] for d in os.listdir(f"{BENCHMARK_DIR}/domains")]

    run_dir = f"{RES_DIR}/run{len(os.listdir(RES_DIR))}"
    os.makedirs(run_dir, exist_ok=True)

    for d in sorted(domains):
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

        dom_dir = f"{run_dir}/{d}"
        os.makedirs(run_dir, exist_ok=True)

        learned_domain = None
        logging.info(f"--------------- Domain {d} ---------------")

        domain_ref_path = f"{BENCHMARK_DIR}/domains/{d}.pddl"
        problems_path = f"{BENCHMARK_DIR}/problems/{d}"
        problems_list = [
            str(p)
            for p in sorted(
                Path(problems_path).glob("*"), key=lambda p: int(p.name.split("_")[0])
            )
        ]

        empty_domain_path = empty_domain(domain_ref_path)
        olam = OLAM(empty_domain_path)
        os.remove(empty_domain_path)

        for problem in problems_list:
            logging.info(f"Running instance {problem}")

            # Set up logging
            prob_dir = f"{dom_dir}/{problem.split('/')[-1].split('.')[0]}"
            os.makedirs(prob_dir, exist_ok=True)
            file_handler = logging.FileHandler(f"{prob_dir}/out.log")
            logging.getLogger().addHandler(file_handler)

            # Create the environment simulator
            sim_problem = PDDLReader().parse_problem(domain_ref_path, problem)
            sim_problem = del_numeric_fluents(sim_problem)

            # Instantiate the environment simulation engine
            start = time.perf_counter()
            simulator = SequentialSimulator(sim_problem)
            learned_domain_str, _ = olam.run(simulator)

            # Close output log file handler
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()

            # Save learned domain in a PDDL file
            domain_learned_path = f"{prob_dir}/domain_learned.pddl"
            with open(domain_learned_path, "w") as f:
                f.write(learned_domain_str)

            # Store syntactic similarity metrics
            metrics = {
                "precision": syntactic_precision(domain_learned_path, domain_ref_path),
                "recall": syntactic_recall(domain_learned_path, domain_ref_path),
                "CPU time": round(time.perf_counter() - start, 2),
            }
            with open(f"{prob_dir}/metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

            # Save the current agent state
            final_state = olam.domain.clone()
            for o in olam.problem.all_objects:
                final_state.add_object(o)
            for fluent in simulator._problem.initial_values:
                final_state.set_initial_value(
                    fluent, simulator._state.get_value(fluent)
                )
            PDDLWriter(final_state).write_problem(f"{prob_dir}/state.pddl")

        logging.info(f"Metrics for domain {d}:\n{pprint.pformat(metrics)}")
