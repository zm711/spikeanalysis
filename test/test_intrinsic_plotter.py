from spikeanalysis.intrinsic_plotter import IntrinsicPlotter
import pytest
import numpy as np
import numpy.testing as nptest


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


def test_generate_amp_bins():
    plotter = IntrinsicPlotter()

    spike_amps = np.array([5.0, 15.0, 19.4, 25.0, 27.0, 30.0])
    probe_len = 100
    pitch = 10
    spike_times = np.array(
        [
            3,
            4,
            5,
            6,
            7,
            8,
        ]
    )
    sp = 1

    depth, amps, dur = plotter._generate_amp_depth_bins(sp, spike_amps, probe_len, pitch, spike_times)

    assert dur == 8
    nptest.assert_array_equal(depth, np.linspace(0, probe_len, num=int(probe_len / pitch)))
    nptest.assert_array_equal(amps, np.linspace(0, 30.0, num=int(30.0 / 30)))


@pytest.fixture
def gab(scope="module"):
    plotter = IntrinsicPlotter()
    spike_amps = np.array([5.0, 15.0, 19.4, 25.0, 27.0, 60.0])
    probe_len = 100
    pitch = 10
    spike_times = np.array(
        [
            3,
            4,
            5,
            6,
            7.0,
            8.0,
        ]
    )
    sp = 1

    depth, amps, dur = plotter._generate_amp_depth_bins(sp, spike_amps, probe_len, pitch, spike_times)

    return depth, amps, dur, spike_amps


def test_compute_cdf_pdf(gab):
    plotter = IntrinsicPlotter()

    depth, amps, dur, spike_amps = gab
    spike_depths = np.array([15, 20, 35, 40, 50.0, 70])
    pdfs, cdfs = plotter._compute_cdf_pdf(spike_amps, spike_depths, amps, depth, dur)

    print("pdf: ", pdfs, "\n")
    print("cdf: ", cdfs)
    nptest.assert_array_equal(pdfs, cdfs)
