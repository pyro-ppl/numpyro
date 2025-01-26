# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


import sys


def get_python_version_specific_seed(
    seed_for_py_3_9: int, seed_not_for_py_3_9: int
) -> int:
    """After release of `jax==0.5.0`, we need different seeds for tests in Python 3.9
    and other versions. This function returns the seed based on the Python version.

    :param seed_for_py_3_9: Seed for Python 3.9
    :param seed_not_for_py_3_9: Seed for other versions of Python
    :return: Seed based on the Python version
    """
    if sys.version_info.minor == 9:
        return seed_for_py_3_9
    else:
        return seed_not_for_py_3_9
