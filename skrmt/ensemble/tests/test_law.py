
import os
import pytest
import shutil
import numpy as np

from skrmt.ensemble import WignerSemicircleDistribution
from skrmt.ensemble import MarchenkoPasturDistribution
from skrmt.ensemble import TracyWidomDistribution
from skrmt.ensemble import ManovaSpectrumDistribution


TMP_DIR_PATH = os.path.join(os.getcwd(), "skrmt/ensemble/tests/tmp")


@pytest.fixture(scope="session", autouse=True)
def _setup_tmp_dir(request):
    """Function that is run before all tests in this script.

    It creates a temporary folder in order to store useful files
    for the following tests.
    """
    # if the directory already exists, it is deleted
    if os.path.exists(TMP_DIR_PATH):
        shutil.rmtree(TMP_DIR_PATH)
    # creating temporary directory  
    os.mkdir(TMP_DIR_PATH)

    # specifying a function that will be run after all tests are completed
    request.addfinalizer(_remove_tmp_dir)


def _remove_tmp_dir():
    """Function that removes the created temporary directory.

    The function is run when all tests in this module are completed.
    """
    shutil.rmtree(TMP_DIR_PATH)



class TestWignerSemicircleDistribution:

    def test_wsd_init_success(self):
        beta = 4
        sigma = 2

        wsd = WignerSemicircleDistribution(beta=beta, sigma=sigma)

        assert wsd.beta == beta
        assert wsd.center == 0.0
        assert wsd.sigma == sigma
        assert wsd.radius == 2.0 * np.sqrt(beta) * sigma
        assert wsd._gaussian_ens is None
    
    def test_wsd_init_raise(self):
        with pytest.raises(ValueError):
            _ = WignerSemicircleDistribution(beta=3)

    def test_wsd_rvs_success(self):
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
        with pytest.raises(ValueError):
            wsd = WignerSemicircleDistribution(beta=1)
            wsd.rvs(-5)
    
    def test_wsd_pdf(self):
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
        fig_name = "test_wsd_pdf_wo_interval.png"
        wsd = WignerSemicircleDistribution()
        wsd.plot_pdf(savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

        fig_name = "test_wsd_pdf_w_interval.png"
        wsd = WignerSemicircleDistribution()
        wsd.plot_pdf(interval=(-2,2), savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wsd_plot_cdf(self):
        fig_name = "test_wsd_cdf_wo_interval.png"
        wsd = WignerSemicircleDistribution()
        wsd.plot_cdf(savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

        fig_name = "test_wsd_cdf_w_interval.png"
        wsd = WignerSemicircleDistribution()
        wsd.plot_cdf(interval=(-2,2), savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

    


