# signlanguage-detector

### How to run..

The repository contains several example pictures and a trained model ( on MRFSegmentation and HoG features).
These are used for the web cam application and the examples. To retrain a model etc. the datasets linked further
down need to be downloaded.

####Required packages
- python3 with numpy, sklearn, [PyMaxFlow](https://github.com/pmneila/PyMaxflow)
- [opencv2](http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv "How to install opencv on ubuntu")

####Runnables
- [app.main](code/app/main.py) runs the web cam application. Optimal conditions are:
    - light source from behind the camera
    - no other moving object/ object with skin colour inside the green box
    - when ready to calibrate move your hand slowly in the box and press "c"
    
- [evaluation.plots.cthresh_asl](code/evaluation/plots/cthresh_asl.py) shows example for colour threshold segmentation on ASL-set
- [evaluation.plots.cthresh_tm](code/evaluation/plots/cthresh_tm) shows example for colour threshold segmentation on TM-set
- [evaluation.plots.mrf_asl](code/evaluation/plots/mrf_asl.py) shows example for Markov Random Field segmentation on ASL-set
- [evaluation.plots.hog_asl](code/evaluation/plots/hog_asl.py) shows example for HoG-descriptor on ASL-set

### Utilities
- *store.codebook* generates Bag-of-HoGs codebook
- *store.descriptors* saves generated descriptors
- *store.models* trains model from data and saves it
- *store.skinhist_asl* generates average histogram of asl set and saves it
- *test.* contains tests for various parts
 
### Datasets
[Fingerspelling-Dataset - ASL](http://empslocal.ex.ac.uk/people/staff/np331/index.php?section=FingerSpellingDataset)

[Fingerspelling Dataset - Thomas Moeslund](http://www-prima.inrialpes.fr/FGnet/data/12-MoeslundGesture/database.html)

### Other
[Overleaf-Report](https://www.overleaf.com/8743838fnnjtvrxbqth#/31192954/)
