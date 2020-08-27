try:
    import haiku
except ImportError:
    raise ImportError("Looking like you want to use flax and/or haiku to declare "
                      "nn modules. This is an experimental feature. "
                      "You need to install `haiku` to be able to use this feature. "
                      "It can be installed with `pip install git+https://github.com/deepmind/dm-haiku`.")
