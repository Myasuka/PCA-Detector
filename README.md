# PCA_Detector
Implementaions of shilling detection algorithm in collaborative filtering, based on breeze, Mahout and Spark respectively

Algorithm adapted in this project comes from paper [Mehta. 2009](http://rd.springer.com/article/10.1007%2Fs11257-008-9050-4), 
the main idea is that using Principal Component Analysis to work on the user-rating matrix, and pick out users which have low 
covariance which are marked as possible spam users.

In general, we use SVD to calculate the main component of the user-rating matrix, which equals to PCA.
