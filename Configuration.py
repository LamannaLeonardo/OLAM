# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


# Output in console or in log file, set to True for debugging
OUTPUT_CONSOLE = False

# For each operator, remove all uncertain negative effects that are not in the preconditions
NEG_EFF_ASSUMPTION = False

# Set resolution time limit
TIME_LIMIT_SECONDS = 3600000

# Set maximum iterations number
MAX_ITER = 10000

# Set planner time limit
PLANNER_TIME_LIMIT = 60

# Maximum negated preconditions length
MAX_PRECS_LENGTH = 8

# Numpy random seed
RANDOM_SEED = 0


#########################################################################################################
# ADL2STRIPS generated file (don't change this)
#########################################################################################################

ADL2STRIPS_FILE = "domain.pddl"


#########################################################################################################
# Test directories
#########################################################################################################

ROOT_DIR = "Analysis/"

ROOT_TEST_DIR = "{}Tests/".format(ROOT_DIR)
ROOT_BENCHMARKS_DIR = "{}Benchmarks/".format(ROOT_DIR)


INSTANCE_DATA_PATH_PDDL = ""  # Updated runtime
BENCHMARK_DIR = ""  # Updated runtime


#########################################################################################################
# PDDL problem files path
#########################################################################################################

DOMAIN_FILE_NAME = "PDDL/domain.pddl"


#########################################################################################################
# LogReader parameters (for statistics)
#########################################################################################################

STRATEGIES = ["FD", "Random"]


#########################################################################################################
# Others
#########################################################################################################
# Java directory containing jdk
JAVA_DIR = 'Java'
# Java bin path
JAVA_BIN_PATH = ""  # This is set runtime
