# Polar Express Project


This repository contains the code for the project Polar Express, from the course in Machine Learning of the Computer Science faculty of UNIBO.

------------------

The aim of the project is as follows: </br>

The purpose of the project is to learn the mapping from polar coordinates to a a discrete 10x10 grid of cells in the plane, using a neural network. 
The supervised dataset is given in the form of a generator (to be considered as a **black box**).

The model must achieve an **accuracy of 95%**, and it will be evaluated in a way **inversely proportional to the number of its parameters: the smaller, the better.**

**WARNING**: *Any solution taking advantage of meta-knowledge about the generator will be automatically rejected*.

------------------

Here is the generator. It returns triples of the form ```((theta,rho),out)``` where ```(theta,rho)``` are the polar coordinates of a point in the first quadrant of the plane, and ```out``` is a 10x10 map with "1" in the cell corresponding to the point position, and "0" everywhere else.
By setting flat=True, the resulting map is flattened into a vector with a single dimension 100. *You can use this variant, if you wish*.

```
def polar_generator(batchsize,grid=(10,10),noise=.002,flat=True):
  while True:
    x = np.random.rand(batchsize)
    y = np.random.rand(batchsize)
    out = np.zeros((batchsize,grid[0],grid[1]))
    xc = (x*grid[0]).astype(int)
    yc = (y*grid[1]).astype(int)
    for b in range(batchsize):
      out[b,xc[b],yc[b]] = 1
    #compute rho and theta and add some noise
    rho = np.sqrt(x**2+y**2) + np.random.normal(scale=noise)
    theta = np.arctan(y/np.maximum(x,.00001)) + np.random.normal(scale=noise)
    if flat:
      out = np.reshape(out,(batchsize,grid[0]*grid[1]))
    yield ((theta,rho),out)
```

------------------

### What to deliver

For the purposes of the project you are supposed to work with the default 10x10 grid, and the default noise=.002
The generator must be treatead as a black box, do not tweak it, and do not exploit its semantics that is supposed to be unknown. You are allowed to work with the "flat" modality, if you prefer so.
You need to:
- define an accuracy function (take inspiration from the code of the previous cell)
- define a neural network taking in input theta and rho, and returning out
- measure the network's accuracy that must be above 95% (accuracy must be evaluated over at least 20000 samples)
- tune the network trying to decrease as much as possible the numer of parameters, preserving an accuracy above 95%. Only your best network must be delivered.


You must deliver a SINGLE notebook working on colab, containing the code of the network, its summary, the training history, the code for the accurary metric and its evaluation on the network.


**N.B. The accuracy must be above 95% but apart from that it does not influence the evaluation. You score will only depend on the number of parameters: the lower, the better.**
