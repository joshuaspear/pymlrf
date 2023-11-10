# Python Machine Learning Research Framework (PyMLRF)

PyMLRF is a general framework to aid with research projects developing and applying machine learning algorithms. The package will implement:
- Classes for handling file systems
- Training and validation loops for Pytorch as well as for more specific frameworks including d3rlpy
- Integration with weights and biases


## Motivation
* When experimenting with ML models there are often a number of artifacts that get reused between models;
* It useful to have configs of these in memory and it is difficult to manage the locations of all such artifacts
* Two base classes have been defined ```FileHandler``` and ```DirectoryHandler``` which tie an object in memory to a disk location

## Model Tracking
* Model tracking is performed using the ```Tracker``` and ```Experiment``` class.
* The ```Tracker``` is an 

