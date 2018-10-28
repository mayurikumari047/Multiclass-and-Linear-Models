# Multiclass-and-Linear-Models
Implementation of Multiclass-to-binary reductions: one-versus-all (OVA), all-versus-all (AVA), and a tree-based reduction (TREE).

## Part 1: Multiclass Classification

In this section, I will explore the differences between three multiclass-to-binary reductions: one-versus-all (OVA), all-versus-all (AVA), and a tree-based reduction (TREE). The evaluation will be on different datasets from  datasets.py.

The classification task I will work with is wine classification. The dataset was downloaded from allwines.com. Job is to predict the type of wine, given the description of the wine. There are two tasks: WineData has 20 different wines, WineDataSmall is just the first five of those (sorted roughly by frequency). Names of the wines can be found in both in WineData.labels as well as the file wines.names.


## Part 2: Gradient Descent and Linear Classification

To get started with linear models, a generic gradient descent method will be implemented. This should go in gd.py, which contains a single (short) function: gd. This takes five parameters: the function we're optimizing, it's gradient, an initial position, a number of iterations to run, and an initial step size.

In each iteration of gradient descent, I will compute the gradient and take a step in that direction, with step size eta. I will use an adaptive step size, where eta is computed as stepSize divided by the square root of the iteration number (counting from one).


## Part 3: Classification with Many Classes

We'll do multiclass classification using Scikit-learn functionality for the Quiz bowl game.

Quiz bowl is a game in which two teams compete head-to-head to answer questions from different areas of knowledge. It lets players interrupt the reading of a question when they know the answer. The goal here is to see how well a classifier performs in predicting the Answer of a question when a different portion of the question is revealed.

