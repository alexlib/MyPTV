## Dumbbell calibration

Source: 
https://openptv-python.readthedocs.io/en/latest/tutorial.html?highlight=dumbbell#dumbbell-calibration


### Reasoning

Sometimes it is inconvenient to position a calibration target. Either
because there is something in the way, or because it is cumbersome to
get the entire target again out of the observation domain. It would be
much easier to move a simple object randomly around the observation
domain and from this perform the calibration.

This is what Dumbbell calibration is doing. The simple
object is a dumbbell with two points separated at a known distance. A
very rough initial guess is sufficient to solve the correspondence
problem for only two particles per image. In other words, the tolerable
epipolar band width is very large: large enough to also find the
correspondence for a very rough calibration, but small enough so as not
to mix up the two points. From there on, calibration optimizes the
distances by which the epipolar lines miss each other, while maintaining
the detected distance of the dumbbell points.

Unlike previous calibration approaches, Dumbbell calibration uses all
camera views simultaneously.

### Inputs:

Somehow, an object with two well visible points has to be moved through the observation domain and recorder. The dumbbells points should be separated by roughly a third of the observation scale.

Note that the accuracy by which these dumbbell points can be determined in 2d, also defines the possible accuracy in 3d.

![image](https://openptv-python.readthedocs.io/en/latest/_images/fig10.png)



## How it should work

1. segmentation of dumbbell images with some specific parameters
2. run a loop over number of images, collect pairs
of points in 3D and check their length, calculate overall error
3. minimize the error by playing with the calibration parameters of all 4 cameras together, 
minimizing the overall error and probably running some multi-variable, slow, gradient descent (that what works for OpenPTV) but without crazy changes, 
possible to do only external parameters first, like 
the single camera calibration approach. 

### Example in PBI

1. segmentation is a bit different if we want 
to have only the pair of points, we can use
template matching of a manually selected dumbbell shape:
https://github.com/yosefm/pbi/blob/master/ptv/dumbbell.py

2. Calibration file correction: 
https://github.com/yosefm/pbi/blob/master/ptv/dumbbell_correct.py
