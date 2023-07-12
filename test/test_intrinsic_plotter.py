from spikeanalysis.intrinsic_plotter import IntrinsicPlotter
import pytest
import numpy as np


def test_IntrinsicPlotter_attributes():
    plotter = IntrinsicPlotter()
    assert plotter.dpi == 800, "dpi is wrong"
    assert plotter.figsize == (10, 8)


def test_IntrinsicPlotter_kwargs():
    plotter = IntrinsicPlotter(**{"dpi": 1200, "x_axis": "Time (ms)"})
    assert plotter.dpi == 1200
    assert plotter.x_axis == "Time (ms)", "check time is used"
    assert plotter.figsize == (10, 8), "fig size should not be changed from default"


def test_sparse_pcs():
    plotter = IntrinsicPlotter()

    pc_feat = np.array(10 * abs(np.random.normal(size=(10, 5, 8))) + 1, dtype=np.int64)
    pc_feat_ind = np.array(10 * abs(np.random.normal(size=(2, 5))) + 1, dtype=np.int64)
    templates = np.array([1, 0, 1, 0, 0, 1, 1, 1, 0, 0])

    print(np.nonzero(pc_feat == 0))
    assert np.shape(np.nonzero(pc_feat == 0)) == (3, 0), "test setup failed. Need pc feat to have no 0s"

    sparse_pc_feat = plotter._sparse_pcs(pc_feat, pc_feat_ind, templates, 3, 4)

    assert np.shape(np.nonzero(sparse_pc_feat == 0))[1] != 0, "sparsifying the pc_feat should incorporate 0s"
    print(np.shape(sparse_pc_feat))
    assert (
        np.shape(sparse_pc_feat)[0] == 10
    ), "should still have 10 spikes"  # 1 index varies since gaussian is not deterministic
