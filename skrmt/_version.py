"""
scikit-rmt version information.
"""
import re

__version__ = "2.0.0"
__version_info__ = tuple(
	int(part)
	for part in re.match(r"(?:\d+!)?(\d+(?:\.\d+)*)", __version__).group(1).split(".")
)
