'''Plot Law Test module

Testing module that plots spectrum limiting laws
'''

import os
import shutil
import pytest

from skrmt.ensemble import wigner_semicircular_law
from skrmt.ensemble import marchenko_pastur_law
from skrmt.ensemble import tracy_widom_law
from skrmt.ensemble import manova_spectrum_distr


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



class TestWignerSemicircleLaw:

    def test_goe_abs_freq(self):
        fig_name = "test_wsl_goe_absfreq.png"
        wigner_semicircular_law(
            ensemble='goe',
            n_size=100,
            bins=100,
            density=False,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_gue_abs_freq(self):
        fig_name = "test_wsl_gue_absfreq.png"
        wigner_semicircular_law(
            ensemble='gue',
            n_size=100,
            bins=100,
            density=False,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_gse_abs_freq(self):
        fig_name = "test_wsl_gse_absfreq.png"
        wigner_semicircular_law(
            ensemble='gse',
            n_size=50,
            bins=100,
            density=False,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_goe_normalized(self):
        fig_name = "test_wsl_goe_norm.png"
        wigner_semicircular_law(
            ensemble='goe',
            n_size=100,
            bins=100,
            density=True,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_gue_normalized(self):
        fig_name = "test_wsl_gue_norm.png"
        wigner_semicircular_law(
            ensemble='gue',
            n_size=100,
            bins=100,
            density=True,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_gse_normalized(self):
        fig_name = "test_wsl_gse_norm.png"
        wigner_semicircular_law(
            ensemble='gse',
            n_size=50,
            bins=100,
            density=True,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_goe_theoretical(self):
        fig_name = "test_wsl_goe_theory.png"
        wigner_semicircular_law(
            ensemble='goe',
            n_size=100,
            bins=100,
            density=True,
            limit_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_gue_theoretical(self):
        fig_name = "test_wsl_gue_theory.png"
        wigner_semicircular_law(
            ensemble='gue',
            n_size=100,
            bins=100,
            density=True,
            limit_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_gse_theoretical(self):
        fig_name = "test_wsl_gse_theory.png"
        wigner_semicircular_law(
            ensemble='gse',
            n_size=50,
            bins=100,
            density=True,
            limit_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wscl_size_exception(self):
        with pytest.raises(ValueError):
            wigner_semicircular_law(
            ensemble='goe',
            n_size=0,
        )
    
    def test_wscl_ensemble_exception(self):
        with pytest.raises(ValueError):
            wigner_semicircular_law(
            ensemble='foo',
        )



class TestMarchenkoPasturLaw():

    def test_wre_abs_freq(self):
        fig_name = "test_mpl_wre_absfreq.png"
        marchenko_pastur_law(
            ensemble='wre',
            p_size=40,
            n_size=100,
            bins=100,
            density=False,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

    def test_wce_abs_freq(self):
        fig_name = "test_mpl_wce_absfreq.png"
        marchenko_pastur_law(
            ensemble='wce',
            p_size=40,
            n_size=100,
            bins=100,
            density=False,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wqe_abs_freq(self):
        fig_name = "test_mpl_wqe_absfreq.png"
        marchenko_pastur_law(
            ensemble='wqe',
            p_size=40,
            n_size=100,
            bins=100,
            density=False,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wre_normalized(self):
        fig_name = "test_mpl_wre_norm.png"
        marchenko_pastur_law(
            ensemble='wre',
            p_size=40,
            n_size=100,
            bins=100,
            density=True,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

    def test_wce_normalized(self):
        fig_name = "test_mpl_wce_norm.png"
        marchenko_pastur_law(
            ensemble='wce',
            p_size=40,
            n_size=100,
            bins=100,
            density=True,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wqe_normalized(self):
        fig_name = "test_mpl_wqe_norm.png"
        marchenko_pastur_law(
            ensemble='wqe',
            p_size=40,
            n_size=100,
            bins=100,
            density=True,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wre_theoretical(self):
        fig_name = "test_mpl_wre_theory.png"
        marchenko_pastur_law(
            ensemble='wre',
            p_size=40,
            n_size=100,
            bins=100,
            density=True,
            limit_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

    def test_wce_theoretical(self):
        fig_name = "test_mpl_wce_theory.png"
        marchenko_pastur_law(
            ensemble='wce',
            p_size=40,
            n_size=100,
            bins=100,
            density=True,
            limit_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wqe_theoretical(self):
        fig_name = "test_mpl_wqe_theory.png"
        marchenko_pastur_law(
            ensemble='wqe',
            p_size=40,
            n_size=100,
            bins=100,
            density=True,
            limit_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wre_ratio_ge1(self):
        fig_name = "test_mpl_wre_ratio_ge1.png"
        marchenko_pastur_law(
            ensemble='wre',
            p_size=200,
            n_size=100,
            bins=100,
            density=False,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wre_theoretical_ratio_ge1(self):
        fig_name = "test_mpl_wre_theory_ratio_ge1.png"
        marchenko_pastur_law(
            ensemble='wre',
            p_size=200,
            n_size=100,
            bins=100,
            density=True,
            limit_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_wre_interval(self):
        fig_name = "test_mpl_wre_interval.png"
        marchenko_pastur_law(
            ensemble='wre',
            p_size=200,
            n_size=100,
            bins=100,
            interval=(0,10),
            density=False,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mpl_size_exception(self):
        with pytest.raises(ValueError):
            marchenko_pastur_law(
            ensemble='mre',
            n_size=0,
        )
    
    def test_mpl_ensemble_exception(self):
        with pytest.raises(ValueError):
            marchenko_pastur_law(
            ensemble='foo',
        )



class TestTracyWidomLaw:

    def test_goe_abs_freq(self):
        fig_name = "test_twl_goe_abs_freq.png"
        tracy_widom_law(
            ensemble='goe',
            n_size=50,
            times=100,
            bins=100,
            density=False,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_gue_abs_freq(self):
        fig_name = "test_twl_gue_abs_freq.png"
        tracy_widom_law(
            ensemble='gue',
            n_size=50,
            times=100,
            bins=100,
            density=False,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_gse_abs_freq(self):
        fig_name = "test_twl_gse_abs_freq.png"
        tracy_widom_law(
            ensemble='gse',
            n_size=25,
            times=100,
            bins=100,
            density=False,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_goe_normalized(self):
        fig_name = "test_twl_goe_normalized.png"
        tracy_widom_law(
            ensemble='goe',
            n_size=50,
            times=100,
            bins=100,
            density=True,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_gue_normalized(self):
        fig_name = "test_twl_gue_normalized.png"
        tracy_widom_law(
            ensemble='gue',
            n_size=50,
            times=100,
            bins=100,
            density=True,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_gse_normalized(self):
        fig_name = "test_twl_gse_normalized.png"
        tracy_widom_law(
            ensemble='gse',
            n_size=25,
            times=100,
            bins=100,
            density=True,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_goe_theoretical(self):
        fig_name = "test_twl_goe_theory.png"
        tracy_widom_law(
            ensemble='goe',
            n_size=50,
            times=100,
            bins=100,
            density=True,
            limit_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_gue_theoretical(self):
        fig_name = "test_twl_gue_theory.png"
        tracy_widom_law(
            ensemble='gue',
            n_size=50,
            times=100,
            bins=100,
            density=True,
            limit_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_gse_theoretical(self):
        fig_name = "test_twl_gse_theory.png"
        tracy_widom_law(
            ensemble='gse',
            n_size=25,
            times=100,
            bins=100,
            density=True,
            limit_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_twl_size_exception(self):
        with pytest.raises(ValueError):
            tracy_widom_law(
            ensemble='goe',
            n_size=0,
        )
    
    def test_twl_ensemble_exception(self):
        with pytest.raises(ValueError):
            tracy_widom_law(
            ensemble='foo',
        )



class TestManovaSpectrumDistr():

    def test_mre_abs_freq(self):
        fig_name = "test_msd_mre_absfreq.png"
        manova_spectrum_distr(
            ensemble='mre',
            m_size=40,
            n1_size=100,
            n2_size=100,
            bins=100,
            density=False,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

    def test_mce_abs_freq(self):
        fig_name = "test_msd_mce_absfreq.png"
        manova_spectrum_distr(
            ensemble='mce',
            m_size=40,
            n1_size=100,
            n2_size=100,
            bins=100,
            density=False,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mqe_abs_freq(self):
        fig_name = "test_msd_mqe_absfreq.png"
        manova_spectrum_distr(
            ensemble='mqe',
            m_size=40,
            n1_size=100,
            n2_size=100,
            bins=100,
            density=False,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mre_normalized(self):
        fig_name = "test_msd_mre_norm.png"
        manova_spectrum_distr(
            ensemble='mre',
            m_size=40,
            n1_size=100,
            n2_size=100,
            bins=100,
            density=True,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

    def test_mce_normalized(self):
        fig_name = "test_msd_mce_norm.png"
        manova_spectrum_distr(
            ensemble='mce',
            m_size=40,
            n1_size=100,
            n2_size=100,
            bins=100,
            density=True,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mqe_normalized(self):
        fig_name = "test_msd_mqe_norm.png"
        manova_spectrum_distr(
            ensemble='mqe',
            m_size=40,
            n1_size=100,
            n2_size=100,
            bins=100,
            density=True,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mre_theoretical(self):
        fig_name = "test_msd_mre_theory.png"
        manova_spectrum_distr(
            ensemble='mre',
            m_size=40,
            n1_size=100,
            n2_size=100,
            bins=100,
            density=True,
            limit_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True

    def test_mce_theoretical(self):
        fig_name = "test_msd_mce_theory.png"
        manova_spectrum_distr(
            ensemble='mce',
            m_size=40,
            n1_size=100,
            n2_size=100,
            bins=100,
            density=True,
            limit_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mqe_theoretical(self):
        fig_name = "test_msd_mqe_theory.png"
        manova_spectrum_distr(
            ensemble='mqe',
            m_size=40,
            n1_size=100,
            n2_size=100,
            bins=100,
            density=True,
            limit_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mre_ratio_ge1(self):
        fig_name = "test_msd_mre_ratio_ge1.png"
        manova_spectrum_distr(
            ensemble='mre',
            m_size=200,
            n1_size=100,
            n2_size=100,
            bins=100,
            density=False,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_msd_mre_theoretical_ratio_ge1(self):
        fig_name = "test_msd_wre_theory_ratio_ge1.png"
        manova_spectrum_distr(
            ensemble='mre',
            m_size=1000,
            n1_size=800,
            n2_size=800,
            bins=100,
            density=True,
            limit_pdf=True,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_mre_interval(self):
        fig_name = "test_msd_mre_interval.png"
        manova_spectrum_distr(
            ensemble='mre',
            m_size=200,
            n1_size=100,
            n2_size=100,
            bins=100,
            interval=(0,10),
            density=False,
            limit_pdf=False,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
    def test_msd_size_exception(self):
        with pytest.raises(ValueError):
            manova_spectrum_distr(
            ensemble='mre',
            m_size=0,
            n1_size=0
        )
    
    def test_msd_ensemble_exception(self):
        with pytest.raises(ValueError):
            manova_spectrum_distr(
            ensemble='foo',
        )
    