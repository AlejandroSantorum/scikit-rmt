"""Utilities for the random matrix ensembles modules
"""

from abc import ABCMeta, abstractmethod


#########################################################################
### ABSTRACT CLASS: ENSEMBLE

class _Ensemble:

    # abstract class
    __metaclass__ = ABCMeta

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def eigval_pdf(self):
        pass