import tomllib
from importlib import resources


def init_config(path=None):
    """Initializes the global configuration dictionary.

    Loads the default package configuration and optionally overrides it
    with values from a user-provided TOML configuration file.

    Args:
        path: Optional path to a TOML configuration file whose values
            override the defaults.
    """
    global _config

    data = {}

    # Load defaults from package
    with resources.files("sfrd").joinpath("default_config.toml").open("rb") as f:
        data.update(tomllib.load(f))

    # Override with user config if provided
    if path:
        with open(path, "rb") as f:
            data.update(tomllib.load(f))

    _config = data


class DictProxy:
    """Thin proxy wrapper around the global configuration dictionary.

    Provides dictionary-style access while allowing runtime mutation for
    testing and temporary overrides.
    """
    def __getitem__(self, key):
        return _config[key]

    # allow changing config at runtime, for testing purposes
    def __setitem__(self, key, value):
        _config[key] = value
    
    def load_config(self, path):
        init_config(path)


init_config()
config = DictProxy()
