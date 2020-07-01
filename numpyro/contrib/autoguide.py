# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import warnings

from numpyro.infer import ELBO as AutoContinuousELBO  # noqa F403
from numpyro.infer.autoguide import *  # noqa F403

warnings.warn("numpyro.contrib.autoguide has moved to numpyro.infer.autoguide. "
              "The contrib alias will stop working in future versions.",
              FutureWarning)
