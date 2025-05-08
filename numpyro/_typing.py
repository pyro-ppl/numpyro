# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from typing_extensions import ParamSpec

P = ParamSpec("P")
ModelT = Callable[P, Any]

Message = dict[str, Any]
TraceT = OrderedDict[str, Message]
