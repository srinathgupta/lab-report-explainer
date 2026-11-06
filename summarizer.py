"""Public entry point for the app and tests.

The logic lives in the `medexplain` package, one module per feature:
    core.py          settings, file checks, Gemini call with retries and fallback
    lab.py           lab reports and report comparison, flag double-check
    scan.py          written scan / test reports
    prescription.py  prescriptions and doctor's notes
    oncology.py      cancer pathology and tumour-marker reports
    followup.py      follow-up questions
"""

import time  # noqa: F401  (tests patch time.sleep through this module)

from medexplain.core import *  # noqa: F401,F403
from medexplain.core import _generate  # noqa: F401
from medexplain.followup import *  # noqa: F401,F403
from medexplain.lab import *  # noqa: F401,F403
from medexplain.oncology import *  # noqa: F401,F403
from medexplain.prescription import *  # noqa: F401,F403
from medexplain.scan import *  # noqa: F401,F403
from medexplain.tracker import *  # noqa: F401,F403
