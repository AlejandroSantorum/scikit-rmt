'''Plot Law Test module

Testing module that plots spectrum limiting laws
'''

import os
import shutil
import pytest

from skrmt.ensemble import wigner_semicircular_law
from skrmt.ensemble import marchenko_pastur_law
from skrmt.ensemble import tracy_widom_law


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
