# Homeworks for Intro to Machine Learning Course

### [Homework 1](https://github.com/rayansami/CS595-intro-to-machine-learning/tree/master/Homework%201)

[Gradient Descent](https://github.com/rayansami/CS595-intro-to-machine-learning/blob/master/Homework%201/gradient-descent.ipynb) : Simply a practice problem for gradient descent. Here I had to develop a dataset using numpy random. Then implemented functions fot cost and gradient. After that I assessed the convergence and observed different behaviour by changing step size and number of iterations

[Multivarate Linear Regression](https://github.com/rayansami/CS595-intro-to-machine-learning/blob/master/Homework%201/multivariate-linear-regression.ipynb) : Solved a polynomial regression problem. Implemented cost function and gradient descent for this problem.

[Linear Regression](https://github.com/rayansami/CS595-intro-to-machine-learning/blob/master/Homework%201/linear-regression.ipynb) : Had a chance to choose dataset and work on that independently. I choosed [bodybrainDataset](https://github.com/rayansami/CS595-intro-to-machine-learning/blob/master/Homework%201/bodybrainDataset.csv) and built a model to predict BodyWeight from BrainWeight.

### [Homework 2](https://github.com/rayansami/CS595-intro-to-machine-learning/tree/master/Homework%202)

[Logistic Regression](https://github.com/rayansami/CS595-intro-to-machine-learning/blob/master/Homework%202/logistic_regression.ipynb) : Used [book data](https://github.com/rayansami/CS595-intro-to-machine-learning/blob/master/Homework%202/book-data.csv) to implement a book classification using logistic regression. Here I predicted whether a book is hardcover or paperback(2 classes). Had to develop(not using libraries) cost function and gradient descent for this problem.

[SMS Classification](https://github.com/rayansami/CS595-intro-to-machine-learning/blob/master/Homework%202/sms_classify.ipynb) : Predicting SMS as spam or ham using logistic regression. Dataset is a text file.

### [Homework 3](https://github.com/rayansami/CS595-intro-to-machine-learning/tree/master/Homework%203)

[Digit Classification](https://github.com/rayansami/CS595-intro-to-machine-learning/blob/master/Homework%203/digit-classification.ipynb) : It's a multiclass classification problem to detect hand-written digits from 0 to 9. Needed to use regularization and showed learning curve.

### [Homework 4](https://github.com/rayansami/CS595-intro-to-machine-learning/tree/master/Homework%204)

[HW Problem Description](https://github.com/rayansami/CS595-intro-to-machine-learning/blob/master/Homework%204/HW4.pdf) 

[K-means Algorithm](https://github.com/rayansami/CS595-intro-to-machine-learning/blob/master/Homework%204/kmeans.py) : Implemented K-means algorithm using data from [A.txt](https://github.com/rayansami/CS595-intro-to-machine-learning/blob/master/Homework%204/A.txt).

It was required to show different choices of K's(ranging from 2 to 10 with step size 1), calculate SSE and plot SSE against K.
![Elbow Curve](./Homework%204/Plot-1-1_Elbow_Curve.png "Elbow Curve")



Run K-means by setting K=3.

![3 Clusters](./Homework%204/Plot-1-2_color_plot_k3.png)


[Hierarchical Clustering](https://github.com/rayansami/CS595-intro-to-machine-learning/blob/master/Homework%204/hierarchical-clustering.py) : Implemented hierarchical clustering using library and showed different results for [MiN Similarity](https://github.com/rayansami/CS595-intro-to-machine-learning/blob/master/Homework%204/Plot-2-1_MIN_Similarity.png),[MAX Similarity](https://github.com/rayansami/CS595-intro-to-machine-learning/blob/master/Homework%204/Plot-2-2_MAX_Similarity.png),[Group Average Similarity](https://github.com/rayansami/CS595-intro-to-machine-learning/blob/master/Homework%204/Plot-2-3_GroupAvg_Similarity.png) and [Centroid Distance Similarity](https://github.com/rayansami/CS595-intro-to-machine-learning/blob/master/Homework%204/Plot-2-4_CentroidDist_Similarity.png) from [B.txt](./Homework%204/B.txt) dataset.

Centroid Distance Similarity

![Centroid Distance Similarity](./Homework%204/Plot-2-4_CentroidDist_Similarity.png)


