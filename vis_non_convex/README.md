# Purpose
Investigate how ML parameters change during training.

# Motivation
Deep learning model optimization is said non-convex,
I wanted to visualise how much non-convex it is.

# Results
Results which are produced using `plot_train.py`, can be seen below.

![Results](Paths.png)

As can be seen on the plots, some parameters have quite a lot of displacement during training.
The direction of movement for these parameters also changes quite a few times during training,
even when using momentum.
As a result parameters can pass through some values multiple times during training.

Running `plot_train.py` will also show the same figure with 3d plot being interactive.