AnalogAnalysis
--------------

This is a class for analyzing data with analog based stimuli. Traditionally stimuli that can follow Gaussian distribution
of values can be assessed with a variety of analyses. Currently this class performs only spike triggered averages, which 
takes each spike and determines what the stimulus around that spike looked like. This can assess if each spike/AP is 
encoding the same information about the stimulus on average.