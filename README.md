# Data_Science-Neural-Network
# About Project:
Implementation of 
* trans regression
* Perceptron
* Neural network with 3 layer 
* Neural network with multiple layer.

Implement forward selection and cross validation on these technique and observe differnce in R square 
adjusted R square and cross validated R square using graph.

# How to run:
### Scalation:
* Go inside Regression folder using terminal
* Run sbt server inisde the Regression folder:     $ sbt
* To compile:     sbt:Regression > compile
* To run:         sbt:Regression > run

Using built in dataset: Unzip the folder then follow the above instruction mentioned step. After run you will have the menu on which dataset you want to run the code. Once you make your choice the code will give you the graph for all the five regression technique mentined above.The graph has three value r^2, adjusted r^2 and cross validated r^2 after applying forward selection.    
 
### Note: 
For other dataset:
The target variable should be the first column of the dataset provided.
Path of the dataset should be provide properly to run the code.


### Python using Keras :
* Download the dataset.
* Download the python file.
* Run it on terminal or jupyter notebook
* Enter the path of the directory where you have stored Dataset
* Choose on which dataset you want  to run
* Choose the type of model you want to run

### Note:
For other dataset:
The target variable should be the first column of the dataset provided.
Path of the dataset should be provide properly to run the code.
and also the column name(feature name)


# DataSet used: 

#### Data set used for Python and Scalation both:
* Auto-Mpg
* Computer Hardware
* Concrete
* Graduate Admission
* Real Estate
* Yacht
* Gps Trajectories
* Beijing PM
* Red wine
* White wine
 
Note: All the data set has been taken from http://archive.ics.uci.edu/ml/datasets.html

# Preprocessing
* we have imputed the mean value in place of 'nan'

# Technology Used:
* R
* Scala
* Scalation
* sbt

# Team Member:
* Priyank Malviya(Priyank.Malviya@uga.edu) 
* Anubhav Nigam (Anubhav.Nigam@uga.edu)
* Anant Tripathi (Anant.Tripathi@uga.edu)

# License
This project is licensed under the MIT License.
