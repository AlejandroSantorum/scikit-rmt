'''Base Ensemble Test module

Testing _Ensemble abstract class
'''

import pytest

from skrmt.ensemble._base_ensemble import _Ensemble


def test_init_exception():
    """Testing the abstract class cannot be instantiated
    """
    with pytest.raises(TypeError):
        _ = _Ensemble()