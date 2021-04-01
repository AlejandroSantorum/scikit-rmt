"""Base Ensemble Module

This module contains the general implementation of the matrix ensembles.
This file contains the common attributes and methods for all the
random matrix ensembles. It also defines the basic interface to be
supported by inherited classes.

"""

from abc import ABCMeta, abstractmethod


#########################################################################
### ABSTRACT CLASS: ENSEMBLE

class _Ensemble:
    """General abstract ensemble class.

    This class contains common attributes and methods for all the
    ensembles. It also defines the basic interface to be
    supported by inherited classes. 

    """

    # abstract class
    __metaclass__ = ABCMeta

    @abstractmethod
    def sample(self):
        """Samples new random matrix.

        The sampling algorithm depends on the inherited classes, so it should be
        specified by them.

        Returns:
            numpy array containing new matrix sampled.
        """
        pass

    @abstractmethod
    def set_size(self):
        pass

    @abstractmethod
    def eigval_pdf(self):
        pass