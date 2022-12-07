'''Tracy-Widom distribution approximator Test module

Testing module that checks Tracy-Widom distribution approximator.
'''

from numpy.testing import assert_almost_equal

from skrmt.ensemble.tracy_widom_approximator import TW_Approximator

def test_cdf_approximation():
    """Testing the approximation of the CDF.
    
    The PDF is not tested since it is the derivative of the CDF.
    """
    tw = TW_Approximator(beta=2)

    assert_almost_equal(tw.cdf(0.1), 0.9754704606594619, decimal=5)

    assert_almost_equal(tw.cdf(-1.0), 0.8072142419992853, decimal=5)
    assert_almost_equal(tw.cdf(-2.0), 0.41322414250512257, decimal=4)
    assert_almost_equal(tw.cdf(-3.0), 0.08031955293933454, decimal=4)
    assert_almost_equal(tw.cdf(-4.0), 0.0035445535955092003, decimal=4)
    assert_almost_equal(tw.cdf(-5.0), 2.135996984741116e-5, decimal=4)
    assert_almost_equal(tw.cdf(-6.0), 1.062254674124451e-8, decimal=4)
    assert_almost_equal(tw.cdf(-7.0), 2.639614767246062e-13, decimal=4)
    assert_almost_equal(tw.cdf(-8.0), 1.9859004257636574e-19, decimal=4)

    assert_almost_equal(tw.cdf(1.0), 0.9975054381493893, decimal=4)
    assert_almost_equal(tw.cdf(2.0), 0.9998875536983092, decimal=4)
    assert_almost_equal(tw.cdf(3.0), 0.9999970059566077, decimal=5)
    assert_almost_equal(tw.cdf(4.0), 0.9999999504208784, decimal=6)
    assert_almost_equal(tw.cdf(5.0), 0.9999999994682207, decimal=7)
    assert_almost_equal(tw.cdf(6.0), 0.9999999999961827, decimal=8)
    assert_almost_equal(tw.cdf(7.0), 0.9999999999999811, decimal=9)
    assert_almost_equal(tw.cdf(8.0), 0.9999999999999999, decimal=10)
