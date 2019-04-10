# The implementation follows the design in PyTorch: torch.distributions.constraint_registry.py
#
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from numpyro.distributions import constraints
from numpyro.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    ExpTransform,
    IdentityTransform,
    SigmoidTransform,
    StickBreakingTransform
)


class ConstraintRegistry(object):
    def __init__(self):
        self._registry = {}

    def register(self, constraint, factory=None):
        if factory is None:
            return lambda factory: self.register(constraint, factory)

        if isinstance(constraint, constraints.Constraint):
            constraint = type(constraint)

        self._registry[constraint] = factory

    def __call__(self, constraint):
        try:
            factory = self._registry[type(constraint)]
        except KeyError:
            raise NotImplementedError

        return factory(constraint)


biject_to = ConstraintRegistry()


@biject_to.register(constraints.greater_than)
def _transform_to_greater_than(constraint):
    return ComposeTransform([ExpTransform(),
                             AffineTransform(constraint.lower_bound, 1,
                                             domain=constraints.positive)])


@biject_to.register(constraints.interval)
def _transform_to_interval(constraint):
    return ComposeTransform([SigmoidTransform(),
                             AffineTransform(constraint.lower_bound, constraint.upper_bound,
                                             domain=constraints.unit_interval)])


@biject_to.register(constraints.real)
def _transform_to_real(constraint):
    return IdentityTransform()


@biject_to.register(constraints.simplex)
def _transform_to_simplex(constraint):
    return StickBreakingTransform()
