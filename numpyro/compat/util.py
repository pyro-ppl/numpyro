# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0


class UnsupportedAPIWarning(Warning):
    """
    Warn users on Pyro operations that do not have a meaningful interpretation
    in NumPyro. Unlike raising NotImplementedError, it might be possible in
    such cases to return a dummy object and recover.
    """
    pass
