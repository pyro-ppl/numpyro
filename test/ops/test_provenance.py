# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from numpyro.ops.provenance import ProvenanceArray, eval_provenance, get_provenance


@pytest.mark.parametrize(
    "f, inputs, expected_output",
    [
        (lambda x, y: x + y, ({"a"}, {}), {"a"}),
        (lambda x, y: x + y, ({"a"}, {"b"}), {"a", "b"}),
        (lambda x, y: x + y, ({"a", "c"}, {"a", "b"}), {"a", "b", "c"}),
        (lambda x, y, z: x + y, ({"a"}, {"b"}, {"c"}), {"a", "b"}),
    ],
)
def test_provenance(f, inputs, expected_output):
    inputs = [ProvenanceArray(np.array([0]), p) for p in inputs]
    assert get_provenance(eval_provenance(f, *inputs)) == expected_output
