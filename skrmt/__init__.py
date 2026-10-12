"""
scikit-rmt package initialization.
"""
import logging

from ._version import __version__, __version_info__

logging.getLogger(__name__).addHandler(logging.NullHandler())
