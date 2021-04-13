"""Python wrapper of UDAR, a part-of-speech tagger for (accented) Russian"""

# NOTE: The order is hierarchical; do not alphabetize!

from .tag import *  # noqa: F401, F403
from .subreading import *  # noqa: F401, F403
from .reading import *  # noqa: F401, F403
from .tok import *  # noqa: F401, F403
from .sentence import *  # noqa: F401, F403
from .document import *  # noqa: F401, F403
from .fsts import *  # noqa: F401, F403

from .convenience import *  # noqa: F401, F403
from .misc import *  # noqa: F401, F403
from .transliterate import *  # noqa: F401, F403

from .features import *  # noqa: F401, F403

from .version import version as __version__
