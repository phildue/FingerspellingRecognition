Implementation
----------------------------------------------------------
- bag-of-words approach with hog for classification
- markov random field for hand segmentation on video

Experiments
----------------------------------------------------------
- Compare descriptors for hand pixel/hog/bag-of-hogs/moments
- Compare classifiers
- Measure improvement by pca

Report
----------------------------------------------------------
- Abstract
- Introduction
    - Summary with Results
    - Focus: What did we implement ourselves?
- Approach
    - Overview
        1. Image from dataset -> Segmentation -> Feature Description -> Training
        2. Image from camera -> Segmentation -> Feature Description -> Classification
    - Dataset
        1. Image segmentation not possible -> Bad results
        2. Easier to segment
    - Image segmentation
        1. Find largest object in binary picture
        2. Background subtraction, image labelling, knn training, markov random field with probabilities
    - Feature Description
        1. Pixel
        2. Moments
        3. histogramm-of-gradients
        4. bag-of-hogs
    - Classification
        1. Dimension reduction
        3. Classifiers
        
- Evaluation
    - Results on Dataset
    - Results in Video
    
- Conclusion