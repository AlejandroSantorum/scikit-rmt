from abc import ABCMeta, abstractmethod
import numpy as np


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


#########################################################################
### Wishart Ensemble = Laguerre Ensemble

class WishartEnsemble(_Ensemble):
    pass

class WishartReal(WishartEnsemble):
    pass

class WishartComplex(WishartEnsemble):
    pass

class WishartQuatern(WishartEnsemble):
    pass


#########################################################################
### Manova Ensemble = Jacobi Ensemble

class ManovaEnsemble(_Ensemble):
    pass

class ManovaReal(ManovaEnsemble):
    pass

class ManovaComplex(ManovaEnsemble):
    pass

class ManovaQuatern(ManovaEnsemble):
    pass


#########################################################################
### Circular Ensemble

class CircularEnsemble(_Ensemble):
    pass

class COE(CircularEnsemble):
    pass

class CUE(CircularEnsemble):
    pass

class CSE(CircularEnsemble):
    pass
