# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpyro import optim


def Adam(kwargs):
    step_size = kwargs.pop("lr")
    b1, b2 = kwargs.pop("betas", (0.9, 0.999))
    eps = kwargs.pop("eps", 1.0e-8)
    return optim.Adam(step_size=step_size, b1=b1, b2=b2, eps=eps)


def ClippedAdam(kwargs):
    step_size = kwargs.pop("lr")
    b1, b2 = kwargs.pop("betas", (0.9, 0.999))
    eps = kwargs.pop("eps", 1.0e-8)
    clip_norm = kwargs.pop("clip_norm", 10.0)
    lrd = kwargs.pop("lrd", None)
    init_lr = step_size
    if lrd is not None:

        def step_size(i):
            return init_lr * lrd ** i

    return optim.ClippedAdam(
        step_size=step_size, b1=b1, b2=b2, eps=eps, clip_norm=clip_norm
    )
