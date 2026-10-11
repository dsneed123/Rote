"""Lets the package run as `python -m rote`; the commands live in cli.py."""

import sys

from .cli import main

sys.exit(main())
