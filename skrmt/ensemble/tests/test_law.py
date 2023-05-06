'''Spectral Law Test module

Testing law module for spectral law simulations
'''

import os
import pytest
import shutil
import numpy as np

from skrmt.ensemble import WignerSemicircleDistribution
from skrmt.ensemble import MarchenkoPasturDistribution
from skrmt.ensemble import TracyWidomDistribution
from skrmt.ensemble import ManovaSpectrumDistribution
from skrmt.ensemble.law import _indicator


TMP_DIR_PATH = os.path.join(os.getcwd(), "skrmt/ensemble/tests/tmp")


@pytest.fixture(scope="module", autouse=True)
def _setup_tmp_dir(request):
    '''Function that is run before all tests in this script.

    It creates a temporary folder in order to store useful files
    for the following tests.
    '''
    # if the directory already exists, it is deleted
    if os.path.exists(TMP_DIR_PATH):
        shutil.rmtree(TMP_DIR_PATH)
    # creating temporary directory  
    os.mkdir(TMP_DIR_PATH)

    # specifying a function that will be run after all tests are completed
    request.addfinalizer(_remove_tmp_dir)


def _remove_tmp_dir():
    '''Function that removes the created temporary directory.

    The function is run when all tests in this module are completed.
    '''
    shutil.rmtree(TMP_DIR_PATH)



class TestWignerSemicircleDistribution:

    def test_wsd_init_success(self):
        '''Testing WignerSemicircleDistribution init
        '''
        beta = 4
        sigma = 2

        wsd = WignerSemicircleDistribution(beta=beta, sigma=sigma)

        assert wsd.beta == beta
        assert wsd.center == 0.0
        assert wsd.sigma == sigma
        assert wsd.radius == 2.0 * np.sqrt(beta) * sigma
        assert wsd._gaussian_ens is None
    
    def test_wsd_init_raise(self):
        '''Testing WignerSemicircleDistribution init raising exception
        '''
        with pytest.raises(ValueError):
            _ = WignerSemicircleDistribution(beta=3)

    def test_wsd_rvs_success(self):
        '''Testing WignerSemicircleDistribution random variates (sampling)
        '''
        beta = 1
        size = 5
        wsd1 = WignerSemicircleDistribution(beta=beta)
        samples = wsd1.rvs(size=size)
        assert len(samples == size)

        beta = 4
        size = 5
        wsd4 = WignerSemicircleDistribution(beta=beta)
        samples = wsd4.rvs(size=size)
        assert len(samples == size)

        size = 10
        samples = wsd4.rvs(size=size)
        assert len(samples == size)
    
    def test_wsd_rvs_raise(self):
        '''Testing WignerSemicircleDistribution random variates (sampling)
        raising an exception because of invalid argument
        '''
        with pytest.raises(ValueError):
            wsd = WignerSemicircleDistribution(beta=1)
            wsd.rvs(-5)
    
    def test_wsd_pdf(self):
        '''Testing WignerSemicircleDistribution pdf
        '''
        beta = 4
        sigma = 2
        center = 0
        wsd = WignerSemicircleDistribution(beta=beta, sigma=sigma, center=center)

        assert wsd.pdf(center) > 0.0
        assert wsd.pdf(center + wsd.radius + 0.1) == 0.0
        assert wsd.pdf(center - wsd.radius - 0.01) == 0.0

        center = 10
        wsd = WignerSemicircleDistribution(beta=beta, sigma=sigma, center=center)

        assert wsd.pdf(center) > 0.0
        assert wsd.pdf(center + wsd.radius + 0.1) == 0.0
        assert wsd.pdf(center - wsd.radius - 0.01) == 0.0
    
    def test_wsd_cdf(self):
        '''Testing WignerSemicircleDistribution cdf
        '''
        beta = 4
        sigma = 2
        center = 0
        wsd = WignerSemicircleDistribution(beta=beta, sigma=sigma, center=center)

        assert wsd.cdf(center) > 0.0
        assert wsd.cdf(center + wsd.radius + 0.1) == 1.0
        assert wsd.cdf(center - wsd.radius - 0.01) == 0.0

        center = 10
        wsd = WignerSemicircleDistribution(beta=beta, sigma=sigma, center=center)

        assert wsd.cdf(center) > 0.0
        assert wsd.cdf(center + wsd.radius + 0.1) == 1.0
        assert wsd.cdf(center - wsd.radius - 0.01) == 0.0
    
    def test_wsd_plot_pdf(self):
        '''Testing WignerSemicircleDistribution plot pdf
        '''
        fig_name = "test_wsd_pdf_wo_interval.png"
        wsd = WignerSemicircleDistribution()
        wsd.plot_pdf(savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

        fig_name = "test_wsd_pdf_w_interval.png"
        wsd = WignerSemicircleDistribution()
        wsd.plot_pdf(interval=(-2,2), savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wsd_plot_cdf(self):
        '''Testing WignerSemicircleDistribution plot cdf
        '''
        fig_name = "test_wsd_cdf_wo_interval.png"
        wsd = WignerSemicircleDistribution()
        wsd.plot_cdf(savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

        fig_name = "test_wsd_cdf_w_interval.png"
        wsd = WignerSemicircleDistribution()
        wsd.plot_cdf(interval=(-2,2), savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wsd_plot_pdf_raise(self):
        '''Testing WignerSemicircleDistribution pdf raising exception
        '''
        with pytest.raises(ValueError):
            wsd = WignerSemicircleDistribution(beta=1)
            wsd.plot_pdf(interval=1)

    

class TestMarchenkoPasturDistribution:

    def test_mpd_init_success(self):
        '''Testing MarchenkoPasturDistribution init
        '''
        beta = 4
        ratio = 1/2
        sigma = 2.0

        mpd = MarchenkoPasturDistribution(beta=beta, ratio=ratio, sigma=sigma)

        assert mpd.beta == beta
        assert mpd.ratio == ratio
        assert mpd.sigma == sigma
        assert mpd.lambda_minus == beta * sigma**2 * (1 - np.sqrt(ratio))**2
        assert mpd.lambda_plus == beta * sigma**2 * (1 + np.sqrt(ratio))**2
        assert mpd._var == beta * sigma**2
        assert mpd._wishart_ens is None
    
    def test_mpd_init_raise(self):
        '''Testing MarchenkoPasturDistribution init raising exception
        '''
        with pytest.raises(ValueError):
            _ = MarchenkoPasturDistribution(ratio=1, beta=3)
        
        with pytest.raises(ValueError):
            _ = MarchenkoPasturDistribution(ratio=0)

    def test_mpd_rvs_success(self):
        '''Testing MarchenkoPasturDistribution random variates (sampling)
        '''
        beta = 1
        ratio = 1/3
        size = 5
        mpd1 = MarchenkoPasturDistribution(beta=beta, ratio=ratio)
        samples = mpd1.rvs(size=size)
        assert len(samples == size)

        beta = 4
        ratio = 1/3
        size = 5
        mpd4 = MarchenkoPasturDistribution(beta=beta, ratio=ratio)
        samples = mpd4.rvs(size=size)
        assert len(samples == size)

        size = 10
        samples = mpd4.rvs(size=size)
        assert len(samples == size)
    
    def test_mpd_rvs_raise(self):
        '''Testing MarchenkoPasturDistribution random variates (sampling) raising
        an exception because of an invalid argument
        '''
        with pytest.raises(ValueError):
            mpd = MarchenkoPasturDistribution(beta=1, ratio=1)
            mpd.rvs(-5)
    
    def test_mpd_pdf(self):
        '''Testing MarchenkoPasturDistribution pdf
        '''
        beta = 4
        ratio = 1/3
        sigma = 1
        mpd = MarchenkoPasturDistribution(beta=beta, ratio=ratio, sigma=sigma)

        middle = np.mean([mpd.lambda_minus, mpd.lambda_plus])
        assert mpd.pdf(middle) > 0.0
        assert mpd.pdf(mpd.lambda_plus + 0.1) == 0.0
        assert mpd.pdf(mpd.lambda_minus - 0.01) == 0.0

        sigma = 10
        mpd = MarchenkoPasturDistribution(beta=beta, ratio=ratio, sigma=sigma)

        middle = np.mean([mpd.lambda_minus, mpd.lambda_plus])
        assert mpd.pdf(middle) > 0.0
        assert mpd.pdf(mpd.lambda_plus + 0.1) == 0.0
        assert mpd.pdf(mpd.lambda_minus - 0.01) == 0.0
    
    def test_mpd_cdf(self):
        '''Testing MarchenkoPasturDistribution cdf
        '''
        beta = 1
        ratio = 1/3
        sigma = 1
        mpd = MarchenkoPasturDistribution(beta=beta, ratio=ratio, sigma=sigma)

        middle = np.mean([mpd.lambda_minus, mpd.lambda_plus])
        assert mpd.cdf(middle) > 0.0
        assert mpd.cdf(mpd.lambda_plus + 0.1) == 1.0
        assert mpd.cdf(mpd.lambda_minus - 0.01) == 0.0

        sigma = 2
        ratio = 2
        mpd = MarchenkoPasturDistribution(beta=beta, ratio=ratio, sigma=sigma)

        middle = np.mean([mpd.lambda_minus, mpd.lambda_plus])
        assert mpd.cdf(middle) > 0.0
        assert mpd.cdf(mpd.lambda_plus + 0.1) == 1.0
        assert mpd.cdf(mpd.lambda_minus - 0.1) == 0.0
    
    def test_mpd_plot_pdf(self):
        '''Testing MarchenkoPasturDistribution plot pdf
        '''
        fig_name = "test_mpd_pdf_wo_interval.png"
        mpd = MarchenkoPasturDistribution(ratio=1/3)
        mpd.plot_pdf(savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

        fig_name = "test_mpd_pdf_w_interval.png"
        mpd = MarchenkoPasturDistribution(ratio=1/3)
        mpd.plot_pdf(interval=(-1,10), savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mpd_plot_cdf(self):
        '''Testing MarchenkoPasturDistribution plot cdf
        '''
        fig_name = "test_mpd_cdf_wo_interval.png"
        mpd = MarchenkoPasturDistribution(ratio=1/3)
        mpd.plot_cdf(savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

        fig_name = "test_mpd_cdf_w_interval.png"
        mpd = MarchenkoPasturDistribution(ratio=1/3)
        mpd.plot_cdf(interval=(-1,10), savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True



class TestTracyWidomDistribution:

    def test_twd_init_success(self):
        '''Testing TracyWidomDistribution init
        '''
        beta = 4

        twd = TracyWidomDistribution(beta=beta)

        assert twd.beta == beta
        assert twd.tw_approx is not None
    
    def test_twd_init_raise(self):
        '''Testing TracyWidomDistribution init raising exception
        '''
        with pytest.raises(ValueError):
            _ = TracyWidomDistribution(beta=3)

    def test_twd_rvs_success(self):
        '''Testing TracyWidomDistribution random variates (sampling)
        '''
        beta = 1
        size = 5
        twd1 = TracyWidomDistribution(beta=beta)
        samples = twd1.rvs(size=size)
        assert len(samples == size)

        beta = 4
        size = 5
        twd4 = TracyWidomDistribution(beta=beta)
        samples = twd4.rvs(size=size)
        assert len(samples == size)

        size = 10
        samples = twd4.rvs(size=size)
        assert len(samples == size)
    
    def test_twd_rvs_raise(self):
        '''Testing TracyWidomDistribution random variates (sampling) raising
        an exception because of an invalid argument
        '''
        with pytest.raises(ValueError):
            twd = TracyWidomDistribution(beta=1)
            twd.rvs(-5)

        with pytest.raises(ValueError):
            twd = TracyWidomDistribution(beta=1)
            twd.rvs(size=5, mtx_size=0)
    
    def test_twd_pdf(self):
        '''Testing TracyWidomDistribution pdf
        '''
        beta = 4
        twd = TracyWidomDistribution(beta=beta)
        assert twd.pdf(-1) > 0.0
        assert twd.pdf(100) < 1e-10
        assert twd.pdf(-100) < 1e-10
    
    def test_twd_cdf(self):
        '''Testing TracyWidomDistribution cdf
        '''
        beta = 4
        twd = TracyWidomDistribution(beta=beta)
        assert twd.cdf(-1) > 0.0
        assert twd.cdf(100) > 0.99999
        assert twd.pdf(-100) < 1e-10
    
    def test_twd_plot_pdf(self):
        '''Testing TracyWidomDistribution plot pdf
        '''
        fig_name = "test_twd_pdf_wo_interval.png"
        twd = TracyWidomDistribution()
        twd.plot_pdf(savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

        fig_name = "test_twd_pdf_w_interval.png"
        twd = TracyWidomDistribution()
        twd.plot_pdf(interval=(-5,5), savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_twd_plot_cdf(self):
        '''Testing TracyWidomDistribution plot cdf
        '''
        fig_name = "test_twd_cdf_wo_interval.png"
        twd = TracyWidomDistribution()
        twd.plot_cdf(savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

        fig_name = "test_twd_cdf_w_interval.png"
        twd = TracyWidomDistribution()
        twd.plot_cdf(interval=(-5,5), savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True



class TestManovaSpectrumDistribution:

    def test_msd_init_success(self):
        '''Testing ManovaSpectrumDistribution init
        '''
        beta = 4
        a = 3
        b = 3

        msd = ManovaSpectrumDistribution(beta=beta, a=a, b=b)

        assert msd.beta == beta
        assert msd.a == a
        assert msd.b == b

        assert msd.lambda_term1 == np.sqrt((a/(a+b)) * (1 - (1/(a+b))))
        assert msd.lambda_term2 == np.sqrt((1/(a+b)) * (1 - (a/(a+b))))
        assert msd.lambda_minus == (msd.lambda_term1 - msd.lambda_term2)**2
        assert msd.lambda_plus == (msd.lambda_term1 + msd.lambda_term2)**2
        assert msd._manova_ens is None
    
    def test_msd_init_raise(self):
        '''Testing ManovaSpectrumDistribution init raising exception
        '''
        with pytest.raises(ValueError):
            _ = ManovaSpectrumDistribution(a=1, b=1, beta=3)
        
        with pytest.raises(ValueError):
            _ = ManovaSpectrumDistribution(a=0, b=0, beta=1)

    def test_msd_rvs_success(self):
        '''Testing ManovaSpectrumDistribution random variates (sampling)
        '''
        beta = 1
        a = b = 3
        size = 5
        msd1 = ManovaSpectrumDistribution(beta=beta, a=a, b=b)
        samples = msd1.rvs(size=size)
        assert len(samples == size)

        beta = 4
        a = b = 3
        size = 5
        msd4 = ManovaSpectrumDistribution(beta=beta, a=a, b=b)
        samples = msd4.rvs(size=size)
        assert len(samples == size)

        size = 10
        samples = msd4.rvs(size=size)
        assert len(samples == size)
    
    def test_msd_rvs_raise(self):
        '''Testing ManovaSpectrumDistribution random variates (sampling) raising an
        exception because of an invalid argument
        '''
        with pytest.raises(ValueError):
            msd = ManovaSpectrumDistribution(beta=1, a=1, b=1)
            msd.rvs(-5)
    
    def test_msd_pdf(self):
        '''Testing ManovaSpectrumDistribution pdf
        '''
        beta = 4
        a = b = 2
        msd = ManovaSpectrumDistribution(beta=beta, a=a, b=b)

        middle = np.mean([msd.lambda_minus, msd.lambda_plus])
        assert msd.pdf(middle) > 0.0
        assert msd.pdf(msd.lambda_plus + 0.1) == 0.0
        assert msd.pdf(msd.lambda_minus - 0.01) == 0.0
    
    def test_msd_cdf(self):
        '''Testing ManovaSpectrumDistribution cdf
        '''
        beta = 4
        a = b = 2
        msd = ManovaSpectrumDistribution(beta=beta, a=a, b=b)

        middle = np.mean([msd.lambda_minus, msd.lambda_plus])
        assert msd.cdf(middle) > 0.0
        assert msd.cdf(msd.lambda_plus + 0.1) == 1.0
        assert msd.cdf(msd.lambda_minus - 0.01) == 0.0
    
    def test_msd_plot_pdf(self):
        '''Testing ManovaSpectrumDistribution plot pdf
        '''
        fig_name = "test_msd_pdf_wo_interval.png"
        msd = ManovaSpectrumDistribution(a=3, b=3)
        msd.plot_pdf(savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

        fig_name = "test_msd_pdf_w_interval.png"
        msd = ManovaSpectrumDistribution(a=3, b=3)
        msd.plot_pdf(interval=(-1,2), savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_msd_plot_cdf(self):
        '''Testing ManovaSpectrumDistribution plot cdf
        '''
        fig_name = "test_msd_cdf_wo_interval.png"
        msd = ManovaSpectrumDistribution(a=3, b=3)
        msd.plot_cdf(savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

        fig_name = "test_msd_cdf_w_interval.png"
        msd = ManovaSpectrumDistribution(a=3, b=3)
        msd.plot_cdf(interval=(-1,2), savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True



def test_indicator_func():
    '''Testing indicator function
    '''
    assert _indicator(1.0, start=1.0, stop=2.0, inclusive="both") == 1.0
    assert _indicator(1.0, start=1.0, stop=2.0, inclusive="left") == 1.0
    assert _indicator(1.0, start=1.0, stop=2.0, inclusive="right") == 0.0
    assert _indicator(1.0, start=1.0, stop=2.0, inclusive="neither") == 0.0
    assert _indicator(2.0, start=1.0, stop=2.0, inclusive="both") == 1.0
    assert _indicator(2.0, start=1.0, stop=2.0, inclusive="left") == 0.0
    assert _indicator(2.0, start=1.0, stop=2.0, inclusive="right") == 1.0
    assert _indicator(2.0, start=1.0, stop=2.0, inclusive="neither") == 0.0
    assert _indicator(2.0, stop=2.0, inclusive="both") == 1.0
    assert _indicator(2.0, stop=2.0, inclusive="left") == 0.0
    assert _indicator(2.0, stop=2.0, inclusive="right") == 1.0
    assert _indicator(2.0, stop=2.0, inclusive="neither") == 0.0

def test_indicator_func_except():
    '''Testing indicator function raising exception
    '''
    with pytest.raises(ValueError):
        _ = _indicator(2.0)
    
    with pytest.raises(ValueError):
        _ = _indicator(2.0, start=2.0, inclusive="foo")
    