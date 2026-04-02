from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="session", autouse=True)
def disable_up_credits():
    from unified_planning.shortcuts import get_environment

    get_environment().credits_stream = None


@pytest.fixture(scope="session")
def benchmarks():
    df = pd.read_excel(Path(__file__).parent / "benchmark.xlsx")
    return df.set_index("Domain").to_dict("index")


@pytest.fixture(scope="session")
def domains(benchmarks):
    """Return list of domains to test."""
    return list(benchmarks.keys())
