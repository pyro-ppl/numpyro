# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from collections.abc import Callable
from typing import Any, TypeAlias

from typing_extensions import ParamSpec

P = ParamSpec("P")
ModelT: TypeAlias = Callable[P, Any]

MessageT: TypeAlias = dict[str, Any]
TraceT: TypeAlias = OrderedDict[str, MessageT]
