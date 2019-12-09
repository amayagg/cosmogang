# Improved WGL
The entire README is a WIP, preliminary draft of FID metrics is below:

## Metrics
### Pixel Intensity
A simple first-order statistic that takes the generated images and the validation set images, flattens them, and then treats the flattened images as "activations" to put in a histogram.

#### Pixel Code
 (metrics.py)
 
### FID
* FID calculates the distance between Inception feature vectors calculated for real and generated images.
    * What's an Inception feature vector? The namesake comes from the Inception network, which is a top-performing image classification model that classifies anything into one of 1,000 classes.
    * Inception feature vectors refer to the activations in the last pooling layer prior to the output of classification.
    
#### FID code
1. Run `calculate_real_stats.py` to produce mu and sigma for the real (validation set) images.
2. Run `get_fid.py` in order to calculate mu and sigma for generated images. With this information the code calculates FID.

 
