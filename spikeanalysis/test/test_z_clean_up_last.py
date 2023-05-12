def test_clean_up():

    import glob
    import os
    
    filename = glob.glob('*.npy')
    
    try:
        for file in filename:
            os.remove(file)
    except:
        assert False, "Cleanup error"