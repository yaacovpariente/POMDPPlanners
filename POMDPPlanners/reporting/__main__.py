# SPDX-License-Identifier: MIT

"""Entry point for ``python -m POMDPPlanners.reporting``."""

import sys

from POMDPPlanners.reporting.cli import main

if __name__ == "__main__":
    sys.exit(main())
