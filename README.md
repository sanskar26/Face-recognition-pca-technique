# Face-recognition-pca-technique
This was my "Summer of code" project on Face recognistion using optimized classical statistical technique Principle Component Analysis.

Checkpoints of the projects:- 
  * Developed a test model to implement Classification and Segregation task on the Iris Flower Data set
  * Used statistical tools like Principal Component Analysis to implement Dimensionality Reduction
  * Designed a fully functional Face Recognition model using MATLAB achieving an accuracy of 97%
  * Integrated a complex Neural Network with the model using Keras library and trained on Yale Face Database

Mechanism of the algorithm:- 
   > RGB colour band images are converted to grayscale which potentially enhances the computation time of the detection process, Image Instances from the dataset was flattened to create a Image Vector space with each row corresponding to each image instance. The data is normalised by subtracting the mean of all the images from each instance to get the feature space enhanced. Covariance matrix of the Image Vector space is calculated, which is then used to calculate the Eigen vectors which are orthogonal vectors along which the variance of the features is maximum. We get N Eigen vector corresponding to each image instance from a Dataset of N images, we then sort the eigen vectors with respect to their Eigen-values, picking m top eigen vectors which highest eigen values. The hyperparameter "m" considers the tradeoff between computation time and better accuracy. 

Milestones:-
  * In my codes for the iris data set which was to get familiar with the concept of PCA, I calculated the mean squared distance and assigned it to the category with minimum mean sqaured distance. Got a accuracy of around 68%.

  * In the advanced data set(i.e yale extended data set) i initially reduced the dimension and proceeded without neural net method which gave an accuracy of 71% but later i realised how the accuracy would be affected by using neural networks so i looked for some reference  and found it on github. the link is-https://github.com/vtshitoyan/simpleNN. I trained the reduced images in NN and then passed the test images through my trained neural net for classification. This time, i got a accuracy about 97%.
