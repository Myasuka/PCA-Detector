# PCA_Detector
Implementations of shilling detection algorithm in collaborative filtering, based on breeze, Mahout and Spark respectively

Algorithm adapted in this project comes from paper [Mehta. 2009](http://rd.springer.com/article/10.1007%2Fs11257-008-9050-4), 
the main idea is that using Principal Component Analysis to work on the user-rating matrix, and pick out users which have low 
covariance which are marked as possible spam users.

In general, we use SVD to calculate the main component of the user-rating matrix, which equals to PCA.

# Usage

As scala, mahout and breeze library may not be included in the runtime, recommended built 
method is `mvn assembly:assembly` to generate assembly jar package.

## PCA with mahout:

1.  use `GenerateData` class on ml-10M data to generate spam user data locally, and upload to HDFS for next step's input.
1.  use `PCAMahout` class to execute PCA in MapReduce and get possible spam users. 
