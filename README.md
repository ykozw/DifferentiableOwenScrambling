# Differentiable Owen Scrambling 

Repository for the paper : Differentiable Owen Scrambling 

```
TBC
```

# Dependancies

No dependancies. The code requires a compiler supporting C++17 (C++11 possible by removing constexpr).

# Building

The project is setup with cmake. The commands to run are the following: 

```
mkdir build
cd build
cmake ..
make -j 8 
```

# Running the code 

Executable have common parameters:

* N: number of samples (replaced by Ns for progressive optimization)
* D: dimenions
* sobol: file to read initial samples from. Does not need to be from the Sobol' sequence or a (0, m, 2)-net in txt file. Only N * D coordinates are read from this file, therefore, at least the dimension should match with the parameter provided. Files can be found in `data/` 
* lr: learning rate
* alpha: smoothing parameter
* lrSchedule: learning rate factor. $lr_{i} = (lrSchedule)^{i} * lr$
* alphaSchedule: alpha factor. $\alpha_{i} = (alphaSchedule)^{i} * \alpha$
* prefix: output directory + prefix. See below for more explaination on outputs of the code
* fill_depth: simulates the bottom of the owen tree by adding small random noise to each points

Loss parameters:

* db: path to intrand database (required for integration error optimization), files are provided in `data/`
* batchSize: batch size for integration error optimization. 
* target: targetPCF
* sigmaGBN: sigma value for GBN. The `actual` gbn factor is $\sigmaGBN * N^{-2/D}$

The scripts outputs multiple file:

* {prefix}_16.dat: points after optimization
* {prefix}_soft.dat: points after optimization where the tree is not binarized
* {prefix}_init.dat: initial points (after the starting random tree is applied, but before optimization)
* {prefix}_soft_tree.txt: floating point values stored in the tree

where {prefix} is the parameter passed to the script

# Plotting points 

A simple python script is provided in order to plot the points. It requires numpy and matplotlib. 

`python plot.py pts.dat`