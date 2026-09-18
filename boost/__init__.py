"""boost: a reusable gradient-boosting framework.

Core knows tabular data, an ordering, a target, a loss and a boosting library.
It never knows what a row means. Projects under projects/<name>/ supply the
data recipe, the feature recipe, model configs and optional evaluation hooks.
"""

__version__ = "0.1.0"
