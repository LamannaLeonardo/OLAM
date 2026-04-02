import unified_planning.environment as up_env
from unified_planning.environment import Environment

# Singleton "hard"
ENV = Environment()

# Rende questo ENV anche il GLOBAL_ENVIRONMENT usato internamente da UP
up_env.GLOBAL_ENVIRONMENT = ENV


def get_env() -> Environment:
    return ENV
