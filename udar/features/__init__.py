from .features import ALL

# Run these modules to get the side effects to happen
from .absolute_length import side_effects  # noqa: F811
from .discourse import side_effects  # noqa: F811
from .lexical_complexity import side_effects  # noqa: F811
from .lexical_familiarity import side_effects  # noqa: F811
from .lexical_variability import side_effects  # noqa: F811
from .morphology import side_effects  # noqa: F811
from .normalized_length import side_effects  # noqa: F811
from .priors import side_effects  # noqa: F811
from .readability import side_effects  # noqa: F811
from .sentence import side_effects  # noqa: F811
from .syntax import side_effects  # noqa: F401, F811

__all__ = ['ALL']
