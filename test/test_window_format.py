from spikeanalysis.utils import verify_window_format



def test_verify_window_one_stim():

    windows = verify_window_format(window=[0,1], num_stim=1)

    assert windows == [[0,1]]

def test_verify_window_multi_stim():

    windows = verify_window_format(window=[0,1], num_stim=3)

    assert len(windows)==3, "did not generate three stimuli"
    assert windows[1][0] == 0, "did not correctly generate the extra stimulus"


def test_verify_window_input_multi_stim():
    windows = verify_window_format(window=[[0,1],[0,2]], num_stim=2)

    assert len(windows)==2
    assert windows[0][0] == 0
    assert windows[1][1] == 2