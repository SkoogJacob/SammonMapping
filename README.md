# Sammon Mapping

[Sammon mapping](https://en.wikipedia.org/wiki/Sammon_mapping) ([original paper here](https://ieeexplore.ieee.org/document/1671271)) 
is a method for reducing the number of dimensions in a dataset to make
it easier to process for the machine, and easier to visualise for humans.

This implementation uses gradient descent to compute each step, deriving from the
derivative formulas found in the original paper.

A simple clustering method is also implemented in `models/BKMeans.py` to be one clustering
technique to employ on the mapped data sets. It compared with some of the provided
clustering methods in SciPy.
