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
from skrmt.ensemble.gaussian_ensemble import GaussianEnsemble
from skrmt.ensemble.misc import indicator


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
        samples = wsd4.rvs(size=size, random_state=1)
        assert len(samples == size)
    
    def test_wsd_rvs_raise(self):
        '''Testing WignerSemicircleDistribution random variates (sampling)
        raising an exception because of invalid argument
        '''
        with pytest.raises(ValueError):
            wsd = WignerSemicircleDistribution(beta=1)
            wsd.rvs(size=-5)
    
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

    def test_wsd_plot_goe_abs_freq(self):
        fig_name = "test_wsl_goe_absfreq.png"
        wsd = WignerSemicircleDistribution(beta=1)
        wsd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wsd_plot_gue_abs_freq(self):
        fig_name = "test_wsl_gue_absfreq.png"
        wsd = WignerSemicircleDistribution(beta=2)
        wsd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wsd_plot_gse_abs_freq(self):
        fig_name = "test_wsl_gse_absfreq.png"
        wsd = WignerSemicircleDistribution(beta=4)
        wsd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wsd_plot_goe_normalized(self):
        fig_name = "test_wsl_goe_norm.png"
        wsd = WignerSemicircleDistribution(beta=1)
        wsd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wsd_plot_gue_normalized(self):
        fig_name = "test_wsl_gue_norm.png"
        wsd = WignerSemicircleDistribution(beta=2)
        wsd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wsd_plot_gse_normalized(self):
        fig_name = "test_wsl_gse_norm.png"
        wsd = WignerSemicircleDistribution(beta=4)
        wsd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wsd_plot_goe_theoretical(self):
        fig_name = "test_wsl_goe_theory.png"
        wsd = WignerSemicircleDistribution(beta=1)
        wsd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wsd_plot_gue_theoretical(self):
        fig_name = "test_wsl_gue_theory.png"
        wsd = WignerSemicircleDistribution(beta=2)
        wsd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wsd_plot_gse_theoretical(self):
        fig_name = "test_wsl_gse_theory.png"
        wsd = WignerSemicircleDistribution(beta=4)
        wsd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wsd_plot_tiny_interval_adjusted(self):
        fig_name = "test_wsl_tiny_interval_adjusted.png"
        wsd = WignerSemicircleDistribution(beta=1)
        wsd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            interval=(-0.1, 0.1),
            density=True,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wsd_plot_size_exception(self):
        with pytest.raises(ValueError):
            wsd = WignerSemicircleDistribution(beta=1)
            wsd.plot_empirical_pdf(sample_size=0)
    
    def test_wsd_plot_ensemble_exception(self):
        with pytest.raises(ValueError):
            wsd = WignerSemicircleDistribution(beta=0)
            wsd.plot_empirical_pdf()
    


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
        samples = mpd4.rvs(size=size, random_state=1)
        assert len(samples == size)
    
    def test_mpd_rvs_raise(self):
        '''Testing MarchenkoPasturDistribution random variates (sampling) raising
        an exception because of an invalid argument
        '''
        with pytest.raises(ValueError):
            mpd = MarchenkoPasturDistribution(beta=1, ratio=1)
            mpd.rvs(size=-5)
    
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

    def test_mpd_plot_wre_abs_freq(self):
        fig_name = "test_mpl_wre_absfreq.png"
        mpd = MarchenkoPasturDistribution(beta=1, ratio=1/3)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name,
            random_state=1,
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

    def test_mpd_plot_wce_abs_freq(self):
        fig_name = "test_mpl_wce_absfreq.png"
        mpd = MarchenkoPasturDistribution(beta=2, ratio=1/3)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name,
            random_state=1,
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mpd_plot_wqe_abs_freq(self):
        fig_name = "test_mpl_wqe_absfreq.png"
        mpd = MarchenkoPasturDistribution(beta=4, ratio=1/3)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name,
            random_state=1,
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mpd_plot_wre_normalized(self):
        fig_name = "test_mpl_wre_norm.png"
        mpd = MarchenkoPasturDistribution(beta=1, ratio=1/3)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name,
            random_state=1,
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

    def test_mpd_plot_wce_normalized(self):
        fig_name = "test_mpl_wce_norm.png"
        mpd = MarchenkoPasturDistribution(beta=2, ratio=1/3)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name,
            random_state=1,
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mpd_plot_wqe_normalized(self):
        fig_name = "test_mpl_wqe_norm.png"
        mpd = MarchenkoPasturDistribution(beta=4, ratio=1/3)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name,
            random_state=1,
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mpd_plot_wre_theoretical(self):
        fig_name = "test_mpl_wre_theory.png"
        mpd = MarchenkoPasturDistribution(beta=1, ratio=1/3)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

    def test_mpd_plot_wce_theoretical(self):
        fig_name = "test_mpl_wce_theory.png"
        mpd = MarchenkoPasturDistribution(beta=2, ratio=1/3)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mpd_plot_wqe_theoretical(self):
        fig_name = "test_mpl_wqe_theory.png"
        mpd = MarchenkoPasturDistribution(beta=4, ratio=1/3)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mpd_plot_wre_ratio_ge1(self):
        fig_name = "test_mpl_wre_ratio_ge1.png"
        mpd = MarchenkoPasturDistribution(beta=1, ratio=1/3)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mpd_plot_wre_theoretical_ratio_ge1(self):
        fig_name = "test_mpl_wre_theory_ratio_ge1.png"
        mpd = MarchenkoPasturDistribution(beta=1, ratio=1/3)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mpd_plot_wre_interval(self):
        fig_name = "test_mpl_wre_interval.png"
        mpd = MarchenkoPasturDistribution(beta=1, ratio=1/3)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mpd_plot_wre_emppdf_sample_size(self):
        fig_name = "test_mpd_wre_emppdf_sample_size.png"
        mpd = MarchenkoPasturDistribution(beta=1, ratio=1/3)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mpd_plot_wce_emppdf_sample_size(self):
        fig_name = "test_mpd_wce_emppdf_sample_size.png"
        mpd = MarchenkoPasturDistribution(beta=2, ratio=1/3)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mpd_plot_wqe_emppdf_sample_size(self):
        fig_name = "test_mpd_wqe_emppdf_sample_size.png"
        mpd = MarchenkoPasturDistribution(beta=4, ratio=1/3)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mpd_plot_wre_emppdf_ratio_g_1(self):
        fig_name = "test_mpd_plot_wre_emppdf_ratio_g_1.png"
        mpd = MarchenkoPasturDistribution(beta=1, ratio=2)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mpd_plot_wre_emppdf_ratio_g_1_density(self):
        fig_name = "test_mpd_plot_wre_emppdf_ratio_g_1_density.png"
        mpd = MarchenkoPasturDistribution(beta=1, ratio=2)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mpd_plot_tiny_interval_adjusted(self):
        fig_name = "test_mpd_plot_tiny_interval_adjusted.png"
        mpd = MarchenkoPasturDistribution(beta=1, ratio=1/2)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            interval=(mpd.lambda_minus+0.1, mpd.lambda_minus-0.1),
            density=True,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mpd_plot_mpl_size_exception(self):
        with pytest.raises(ValueError):
            mpd = MarchenkoPasturDistribution(beta=1, ratio=1/3)
            mpd.plot_empirical_pdf(sample_size=0)
    
    def test_mpd_plot_mpl_ensemble_exception(self):
        with pytest.raises(ValueError):
            mpd = MarchenkoPasturDistribution(beta=0, ratio=1/3)
            mpd.plot_empirical_pdf()



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
        samples = twd4.rvs(size=size, random_state=1)
        assert len(samples == size)
    
    def test_twd_rvs_raise(self):
        '''Testing TracyWidomDistribution random variates (sampling) raising
        an exception because of an invalid argument
        '''
        with pytest.raises(ValueError):
            twd = TracyWidomDistribution(beta=1)
            twd.rvs(size=-5)
    
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
    
    def test_twd_plot_goe_abs_freq(self):
        fig_name = "test_twl_goe_abs_freq.png"
        twd = TracyWidomDistribution(beta=1)
        twd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_twd_plot_gue_abs_freq(self):
        fig_name = "test_twl_gue_abs_freq.png"
        twd = TracyWidomDistribution(beta=2)
        twd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_twd_plot_gse_abs_freq(self):
        fig_name = "test_twl_gse_abs_freq.png"
        twd = TracyWidomDistribution(beta=4)
        twd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_twd_plot_goe_normalized(self):
        fig_name = "test_twl_goe_normalized.png"
        twd = TracyWidomDistribution(beta=1)
        twd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_twd_plot_gue_normalized(self):
        fig_name = "test_twl_gue_normalized.png"
        twd = TracyWidomDistribution(beta=2)
        twd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_twd_plot_gse_normalized(self):
        fig_name = "test_twl_gse_normalized.png"
        twd = TracyWidomDistribution(beta=4)
        twd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_twd_plot_goe_theoretical(self):
        fig_name = "test_twl_goe_theory.png"
        twd = TracyWidomDistribution(beta=1)
        twd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_twd_plot_gue_theoretical(self):
        fig_name = "test_twl_gue_theory.png"
        twd = TracyWidomDistribution(beta=2)
        twd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_twd_plot_gse_theoretical(self):
        fig_name = "test_twl_gse_theory.png"
        twd = TracyWidomDistribution(beta=4)
        twd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_twd_plot_tiny_interval_adjusted(self):
        fig_name = "test_twd_plot_tiny_interval_adjusted.png"
        twd = TracyWidomDistribution(beta=1)
        twd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            interval=(-0.2, -0.1),
            density=True,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_twd_plot_ensemble_max_eigvals(self):
        '''Testing plotting max eigenvalues histogram of an ensemble and comparing it
        with Tracy-Widom distribution
        '''
        fig_name = "test_twd_plot_ensemble_max_eigvals.png"
        beta = 1

        twd = TracyWidomDistribution(beta=beta)
        ens = GaussianEnsemble(beta=beta, n=10)
        twd.plot_ensemble_max_eigvals(
            ensemble=ens, n_eigvals=1, bins=10, random_state=1, savefig_path=TMP_DIR_PATH+"/"+fig_name,
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

        fig_name2 = "test_twd_plot_ensemble_max_eigvals_other_beta.png"
        twd = TracyWidomDistribution(beta=1)
        ens2 = GaussianEnsemble(beta=2, n=10)  # note beta is different than TWD instance
        twd.plot_ensemble_max_eigvals(
            ensemble=ens2, n_eigvals=1, bins=10, random_state=1, savefig_path=TMP_DIR_PATH+"/"+fig_name2,
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name2)) == True
    
    def test_twd_plot_size_exception(self):
        with pytest.raises(ValueError):
            twd = TracyWidomDistribution(beta=1)
            twd.plot_empirical_pdf(sample_size=0)
    
    def test_twd_plot_ensemble_exception(self):
        with pytest.raises(ValueError):
            twd = TracyWidomDistribution(beta=0)
            twd.plot_empirical_pdf(sample_size=10)



class TestManovaSpectrumDistribution:

    def test_msd_init_success(self):
        '''Testing ManovaSpectrumDistribution init
        '''
        beta = 4
        a = 3
        b = 3

        msd = ManovaSpectrumDistribution(beta=beta, ratio_a=a, ratio_b=b)

        assert msd.beta == beta
        assert msd.ratio_a == a
        assert msd.ratio_b == b

        assert msd.lambda_term1 == np.sqrt((a/(a+b)) * (1 - (1/(a+b))))
        assert msd.lambda_term2 == np.sqrt((1/(a+b)) * (1 - (a/(a+b))))
        assert msd.lambda_minus == (msd.lambda_term1 - msd.lambda_term2)**2
        assert msd.lambda_plus == (msd.lambda_term1 + msd.lambda_term2)**2
    
    def test_msd_init_raise(self):
        '''Testing ManovaSpectrumDistribution init raising exception
        '''
        with pytest.raises(ValueError):
            _ = ManovaSpectrumDistribution(ratio_a=1, ratio_b=1, beta=3)
        
        with pytest.raises(ValueError):
            _ = ManovaSpectrumDistribution(ratio_a=0, ratio_b=0, beta=1)

    def test_msd_rvs_success(self):
        '''Testing ManovaSpectrumDistribution random variates (sampling)
        '''
        beta = 1
        a = b = 3
        size = 5
        msd1 = ManovaSpectrumDistribution(beta=beta, ratio_a=a, ratio_b=b)
        samples = msd1.rvs(size=size)
        assert len(samples == size)

        beta = 4
        a = b = 3
        size = 5
        msd4 = ManovaSpectrumDistribution(beta=beta, ratio_a=a, ratio_b=b)
        samples = msd4.rvs(size=size)
        assert len(samples == size)

        size = 10
        samples = msd4.rvs(size=size, random_state=1)
        assert len(samples == size)
    
    def test_msd_rvs_raise(self):
        '''Testing ManovaSpectrumDistribution random variates (sampling) raising an
        exception because of an invalid argument
        '''
        with pytest.raises(ValueError):
            msd = ManovaSpectrumDistribution(beta=1, ratio_a=1, ratio_b=1)
            msd.rvs(size=-5)
    
    def test_msd_pdf(self):
        '''Testing ManovaSpectrumDistribution pdf
        '''
        beta = 4
        a = b = 2
        msd = ManovaSpectrumDistribution(beta=beta, ratio_a=a, ratio_b=b)

        middle = np.mean([msd.lambda_minus, msd.lambda_plus])
        assert msd.pdf(middle) > 0.0
        assert msd.pdf(msd.lambda_plus + 0.1) == 0.0
        assert msd.pdf(msd.lambda_minus - 0.01) == 0.0
    
    def test_msd_cdf(self):
        '''Testing ManovaSpectrumDistribution cdf
        '''
        beta = 4
        a = b = 2
        msd = ManovaSpectrumDistribution(beta=beta, ratio_a=a, ratio_b=b)

        middle = np.mean([msd.lambda_minus, msd.lambda_plus])
        assert msd.cdf(middle) > 0.0
        assert msd.cdf(msd.lambda_plus + 0.1) == 1.0
        assert msd.cdf(msd.lambda_minus - 0.01) == 0.0
    
    def test_msd_plot_pdf(self):
        '''Testing ManovaSpectrumDistribution plot pdf
        '''
        fig_name = "test_msd_pdf_wo_interval.png"
        msd = ManovaSpectrumDistribution(ratio_a=3, ratio_b=3)
        msd.plot_pdf(savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

        fig_name = "test_msd_pdf_w_interval.png"
        msd = ManovaSpectrumDistribution(ratio_a=3, ratio_b=3)
        msd.plot_pdf(interval=(-1,2), savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_msd_plot_cdf(self):
        '''Testing ManovaSpectrumDistribution plot cdf
        '''
        fig_name = "test_msd_cdf_wo_interval.png"
        msd = ManovaSpectrumDistribution(ratio_a=3, ratio_b=3)
        msd.plot_cdf(savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

        fig_name = "test_msd_cdf_w_interval.png"
        msd = ManovaSpectrumDistribution(ratio_a=3, ratio_b=3)
        msd.plot_cdf(interval=(-1,2), savefig_path=TMP_DIR_PATH+"/"+fig_name)
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_msd_plot_mre_abs_freq(self):
        fig_name = "test_msd_mre_absfreq.png"
        msd = ManovaSpectrumDistribution(beta=1, ratio_a=3, ratio_b=3)
        msd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name,
            random_state=1,
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

    def test_msd_plot_mce_abs_freq(self):
        fig_name = "test_msd_mce_absfreq.png"
        msd = ManovaSpectrumDistribution(beta=2, ratio_a=3, ratio_b=3)
        msd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name,
            random_state=1,
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_msd_plot_mqe_abs_freq(self):
        fig_name = "test_msd_mqe_absfreq.png"
        msd = ManovaSpectrumDistribution(beta=4, ratio_a=3, ratio_b=3)
        msd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name,
            random_state=1,
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_msd_plot_mre_normalized(self):
        fig_name = "test_msd_mre_norm.png"
        msd = ManovaSpectrumDistribution(beta=1,ratio_a=3, ratio_b=3)
        msd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name,
            random_state=1,
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

    def test_msd_plot_mce_normalized(self):
        fig_name = "test_msd_mce_norm.png"
        msd = ManovaSpectrumDistribution(beta=2, ratio_a=3, ratio_b=3)
        msd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name,
            random_state=1,
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_msd_plot_mqe_normalized(self):
        fig_name = "test_msd_mqe_norm.png"
        msd = ManovaSpectrumDistribution(beta=4, ratio_a=3, ratio_b=3)
        msd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name,
            random_state=1,
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_msd_plot_mre_theoretical(self):
        fig_name = "test_msd_mre_theory.png"
        msd = ManovaSpectrumDistribution(beta=1, ratio_a=3, ratio_b=3)
        msd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

    def test_msd_plot_mce_theoretical(self):
        fig_name = "test_msd_mce_theory.png"
        msd = ManovaSpectrumDistribution(beta=2, ratio_a=3, ratio_b=3)
        msd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_msd_plot_mqe_theoretical(self):
        fig_name = "test_msd_mqe_theory.png"
        msd = ManovaSpectrumDistribution(beta=4, ratio_a=3, ratio_b=3)
        msd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_msd_plot_mre_ratio_ge1(self):
        fig_name = "test_msd_mre_ratio_ge1.png"
        msd = ManovaSpectrumDistribution(beta=1, ratio_a=3, ratio_b=3)
        msd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_msd_plot_msd_mre_theoretical_ratio_ge1(self):
        fig_name = "test_msd_wre_theory_ratio_ge1.png"
        msd = ManovaSpectrumDistribution(beta=1, ratio_a=3, ratio_b=3)
        msd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            density=True,
            plot_law_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_msd_plot_mre_interval(self):
        fig_name = "test_msd_mre_interval.png"
        msd = ManovaSpectrumDistribution(beta=1, ratio_a=3, ratio_b=3)
        msd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            interval=(0,10),
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_msd_plot_mre_a_le_1_b_le_1(self):
        fig_name = "test_msd_plot_mre_a_le_1_b_le_1.png"
        msd = ManovaSpectrumDistribution(beta=1, ratio_a=0.9, ratio_b=0.9)
        msd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            interval=(0,10),
            density=False,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_msd_plot_mre_a_le_1_b_le_1_density(self):
        fig_name = "test_msd_plot_mre_a_le_1_b_le_1_density.png"
        msd = ManovaSpectrumDistribution(beta=1, ratio_a=0.9, ratio_b=0.9)
        msd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            interval=(0,10),
            density=True,
            plot_law_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

    def test_msd_plot_tiny_interval_adjusted(self):
        fig_name = "test_msd_plot_tiny_interval_adjusted.png"
        mpd = ManovaSpectrumDistribution(beta=1, ratio_a=2, ratio_b=2)
        mpd.plot_empirical_pdf(
            sample_size=10,
            bins=10,
            interval=(mpd.lambda_minus+0.1, mpd.lambda_minus-0.1),
            density=True,
            plot_law_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

    def test_msd_plot_size_exception(self):
        with pytest.raises(ValueError):
            msd = ManovaSpectrumDistribution(beta=1, ratio_a=3, ratio_b=3)
            msd.plot_empirical_pdf(sample_size=0)
    
    def test_msd_plot_ensemble_exception(self):
        with pytest.raises(ValueError):
            msd = ManovaSpectrumDistribution(beta=0, ratio_a=3, ratio_b=3)
            msd.plot_empirical_pdf(sample_size=10)



def test_indicator_func():
    '''Testing indicator function
    '''
    assert indicator(1.0, start=1.0, stop=2.0, inclusive="both") == 1.0
    assert indicator(1.0, start=1.0, stop=2.0, inclusive="left") == 1.0
    assert indicator(1.0, start=1.0, stop=2.0, inclusive="right") == 0.0
    assert indicator(1.0, start=1.0, stop=2.0, inclusive="neither") == 0.0
    assert indicator(2.0, start=1.0, stop=2.0, inclusive="both") == 1.0
    assert indicator(2.0, start=1.0, stop=2.0, inclusive="left") == 0.0
    assert indicator(2.0, start=1.0, stop=2.0, inclusive="right") == 1.0
    assert indicator(2.0, start=1.0, stop=2.0, inclusive="neither") == 0.0
    assert indicator(2.0, stop=2.0, inclusive="both") == 1.0
    assert indicator(2.0, stop=2.0, inclusive="left") == 0.0
    assert indicator(2.0, stop=2.0, inclusive="right") == 1.0
    assert indicator(2.0, stop=2.0, inclusive="neither") == 0.0

def test_indicator_func_except():
    '''Testing indicator function raising exception
    '''
    with pytest.raises(ValueError):
        _ = indicator(2.0)
    
    with pytest.raises(ValueError):
        _ = indicator(2.0, start=2.0, inclusive="foo")
    