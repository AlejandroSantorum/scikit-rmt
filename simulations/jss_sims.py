import os
import sys
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)
sys.path.append(dirname[:dirname.rfind("/")])  # TODO: Remove

from skrmt.ensemble.gaussian_ensemble import GaussianEnsemble
from skrmt.ensemble.wishart_ensemble import WishartEnsemble
from skrmt.ensemble.manova_ensemble import ManovaEnsemble
from skrmt.ensemble.circular_ensemble import CircularEnsemble
from skrmt.ensemble.spectral_law import (
    WignerSemicircleDistribution,
    MarchenkoPasturDistribution,
    TracyWidomDistribution,
    ManovaSpectrumDistribution,
)
from skrmt.ensemble.utils import standard_vs_tridiag_hist

IMGS_DIRNAME = "skrmt_sim_imgs"
SCRIPT_PATH = os.path.dirname(__file__)


def setup_img_dir():
    """Creates a directory (if ti does not exist) in the same locations
    as this script for the simulation images of scikit-rmt.
    """
    # creating a directory in the same place as this script for the images
    imgs_dir_path = os.path.join(SCRIPT_PATH, IMGS_DIRNAME)
    if not os.path.exists(imgs_dir_path):
        print(f"Creating directory {IMGS_DIRNAME} at {SCRIPT_PATH} for the images.")
        os.makedirs(imgs_dir_path)

    print(f"The images of the Figures will be stored at {imgs_dir_path}")


def plot_figure_1():
    print("Generating images of Figure 1. This may take some seconds...")

    ens = GaussianEnsemble(beta=1, n=1000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig1_goe_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    ens = GaussianEnsemble(beta=2, n=1000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig1_gue_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    ens = GaussianEnsemble(beta=4, n=1000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig1_gse_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    print("[DONE] - Images Figure 1")


def plot_figure_2():
    print("Generating images of Figure 2. This may take some seconds...")

    ens = WishartEnsemble(beta=1, p=1000, n=5000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig2_wre_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    ens = WishartEnsemble(beta=2, p=1000, n=5000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig2_wce_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    ens = WishartEnsemble(beta=4, p=1000, n=5000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig2_wqe_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    print("[DONE] - Images Figure 2")


def plot_figure_3():
    print("Generating images of Figure 3. This may take some seconds...")

    ens = ManovaEnsemble(beta=1, m=1000, n1=2000, n2=2000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig3_mre_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    ens = ManovaEnsemble(beta=2, m=1000, n1=2000, n2=2000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig3_mce_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    ens = ManovaEnsemble(beta=4, m=1000, n1=2000, n2=2000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig3_mqe_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    print("[DONE] - Images Figure 3")


def plot_figure_4():
    print("Generating images of Figure 4. This may take some seconds...")

    ens = CircularEnsemble(beta=1, n=1000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig4_coe_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    ens = CircularEnsemble(beta=2, n=1000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig4_cue_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    ens = CircularEnsemble(beta=4, n=1000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig4_cse_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    print("[DONE] - Images Figure 4")


def plot_figure_5():
    print("Generating images of Figure 5. This may take some seconds...")

    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig5_goe_1000.png")
    ens = GaussianEnsemble(beta=1, n=1000)
    standard_vs_tridiag_hist(ensemble=ens, bins=60, random_state=10, savefig_path=ens_figpath)

    print("[DONE] - Images Figure 5")


# TODO: Function for Figure 6 !!!


def plot_figure_7():
    pass



def main():
    setup_img_dir()
    # plot_figure_1()
    # plot_figure_2()
    # plot_figure_3()
    # plot_figure_4()

    # plot_figure_5()

    # TODO: Figure 6 !!!!

    plot_figure_7()



if __name__ == "__main__":
    main()