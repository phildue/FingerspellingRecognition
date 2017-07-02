# signlanguage-detector

### How to run..

####Required packages
- python3 with numpy, sklearn, [PyMaxFlow](https://github.com/pmneila/PyMaxflow)
- [opencv2](http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv)

####Runnables
- app.main runs the web cam application. Instructions are displayed via terminal.
- evaluation.plots.cthresh_asl shows example for colour threshold segmentation on ASL-set
- evaluation.plots.cthresh_tm shows example for colour threshold segmentation on TM-set
- evaluation.plots.mrf_asl shows example for Markov Random Field segmentation on ASL-set
- evaluation.plots.hog_asl shows example for HoG-descriptor on ASL-set

### Utilities
- store.codebook generates Bag-of-HoGs codebook
- store.descriptors saves generated descriptors
- store.models trains model from data and saves it
- store.skinhist_asl generates average histogram of asl set and saves it
- test.* contains tests for various parts
 
### Datasets
[Fingerspelling-Dataset - ASL](http://empslocal.ex.ac.uk/people/staff/np331/index.php?section=FingerSpellingDataset)

[Fingerspelling Dataset - Thomas Moeslund](http://www-prima.inrialpes.fr/FGnet/data/12-MoeslundGesture/database.html)

### Other
[Overleaf-Report](https://www.overleaf.com/8743838fnnjtvrxbqth#/31192954/)
