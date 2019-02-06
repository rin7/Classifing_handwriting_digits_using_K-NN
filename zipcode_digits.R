
#implement the k-nearest neighbors algorithm for classifying handwritten digits in
#zip codes which contain the numbers 0 through 9

library(dplyr)
library(tidyr)
library(ggplot2)

#Fuction reading data in files
######################################################################
read_digits <- function(filename) {
  
  data <- read.table(filename)
  
  colnames(data)[1] <- "digit"
  data$digit <- factor(data$digit)
  
  data

}
  
training_set <- read_digits("train.txt")
test_set <- read_digits("test.txt")


#function to view a single digit

view_digit = function(data, obs) {
  
  a = data[,-1]     #removes 1's column (with a 'digit')
  #arranging by column is faster in r
   m = matrix(unlist(a[obs,]), 16,16)  
   
  #function to rotate the matrix
  #https://stackoverflow.com/questions/16496210/rotate-a-matrix-in-r
  #an alternative would be to swap the limits of y-axis when displaying
     rotate = function(x) t(apply(x, 1, rev))   
  
  #displays in a viewer friendly format: 
  #black digit on white background  
  image(rotate((-1)*m),col=paste("gray",1:99,sep=""))           
}

view_digit(training_set, 4)


# Means of digits


#get rows that have digit i
y = NULL       #allocate space
for(i in 0:9) {
  y[[i+1]] = training_set[training_set$digit %in% i,]
}

par(mfrow = c(2, 5))          #make the 10 digit images to fit onto 1 image. 2 row, 5 column
x = matrix(numeric(10*257), nrow = 10, ncol = 257)               #allocate space
for(j in 1:10){
  for(i in 1:257){x[j,i] = mean(y[[j]][,i])}             #average the pixel values for each entry
  view_digit(x, j)                 #create grayscale images
}

#convert x into dataframe so that sapply works
j = data.frame(x)      

variance = sapply(j, var)        #calculate variance
sort(variance)            #order variance
order(variance)           #see which pixels have most/least variance
matrix(c(2:257), nrow = 16, ncol = 16, byrow = TRUE)         #look at where pixels are in image

matrix(order(order(variance[2:257])), nrow = 16, ncol = 16, byrow = TRUE)          #see where are the pixels that have most/least variance


#function to predict knn
########################################################################
#this function take as argument any vector of points from either training 
#or the test set, then computes the appropriate distance matrix
#only between the selected points and the ones of the training set
#however, for any method, the distance matrix is computed only ones
#Note that the first column of the training set is passed into it as labels

predict_knn = function(points, train, labels, k, method) {
  
  n = nrow(train)
  m = nrow(points)
  
  combined = rbind(train, points)
  mt_combined = as.matrix(combined)
  
  #distance matrix computed within the function
  #but only once and only between the points selected and
  #the observations of the training set.
  #a single point or any combination of points from 
  #any data set that user selects
  distance_matrix = dist(mt_combined,
                         method = method,
                         diag = T, upper = T)
  
  distance_matrix <- as.matrix(distance_matrix)
  
  
  for (ind in split_indexes) {
    distance_matrix[ind,ind] <- Inf
  }
  
  f = function(col) {
    v = head(order(col), k)
    t = table(labels[c(v)])
    # browser()
    result <- names(t)[which(t == max(t))]
    
    if(length(result) > 1)
      result <- sample(result, 1)
    result
  }
  
  predictions <- apply(distance_matrix, 2, f)
  names(predictions) <- NULL
  predictions
}

#predicts five nearest neighbors for the first five values of the test set.
predict_knn(test_set[1:5,-1], training_set[,-1], training_set[,1], 5, "euclidean")


#cv_erro_knn()
#we use a helper function: compute_distances(), which takes as arguments
#a data set for which we running the cv, in our case a train, and the desired method
#The cv_error_knn() in turn accepts the resulting distance matrix
#as an argument and performs all the stepps of the cross validation.
# * we were okayed by the TA's on using a for-loop to run this function
# for each value of k= 1:15, w/o significant loss of efficiency (since, there
# is no repeated distance matrix computations involved. However, if the array of k's was 
#larager than that, we would have used an sapply on the vector of k's

set.seed(123) 

#splitting and shuffling the indeces of the training set for 10-fold cross validation
#https://stackoverflow.com/questions/3318333/split-a-vector-into-chunks-in-r
split_indexes <- split(indexes, ceiling(seq_along(indexes)/(n/10)))

#function to compute distance matrix
compute_distances = function(train, method) {
  
  mt_train = as.matrix(train)
  
  distance_matrix = dist(mt_train,
                         method = method,
                         diag = T, upper = T)
  
  distance_matrix <- as.matrix(distance_matrix)
  
  distance_matrix
  
}

#computing the 3 distance matrices
distance_matrix_euc = compute_distances(training_set[,-1], "euclidean")
distance_matrix_mink = compute_distances(training_set[,-1], "minkowski")
distance_matrix_manh = compute_distances(training_set[,-1], "manhattan")

#function to compute error rates for 10-fold cross validation
cv_error_knn = function(distance_matrix, labels, k) {

  #setting the distance from a point to itself to infinity for 
  # all the points. This takes care of 
  # the overlaping observations. (per Ben's OH.) 
  for (ind in split_indexes) {
    distance_matrix[ind,ind] <- Inf
  }
  
  predictions = apply(distance_matrix, 2, function(col) {
    
    v = head(order(col), k)
    t = table(training_set[v,1])
    # print(t)
    result <- names(t)[which(t == max(t))]
    
    if(length(result) > 1)
      result <- sample(result, 1)
    factor(result)
  } )
  # browser()
   names(predictions) <- NULL  
   mean(predictions != labels) 
   #using mean function in the statement above is equivalent to summing the incorrct predictoins
   #and dividing by the total number of predictions.
 
  
}

#setting the k's for which to cross validate
k = c(1:15)

#pre-allocating memory for the vector of predicted error rates
error_rates_euc = k

#computing error rates for euclidian method
for (i in seq_along(k)) {
  
  error_rates_euc[i] = cv_error_knn(distance_matrix_euc, training_set[,1], i)  
  error_rates_euc[i]
}

#pre-allocating memory for the vector of predicted error rates
error_rates_manh = k

##computing error rates for manhattan method
for (i in seq_along(k)) {
  
  error_rates_manh[i] = cv_error_knn(distance_matrix_manh, training_set[,1], i)  
  error_rates_manh[i]
}



# Plotting 10-fold cv error rates


#combining the results into a dataframe for easy display

errors_rate <- data.frame(k,error_rates_euc, error_rates_manh)

#Note that if we were to perform any analysis on this dataframe
#we would tidy it appropriately to make sure that each column 
#is a variabl (here the variables would be: k, method, and error rate 
#(see the commented out code at the end of the question)

p <- ggplot(errors_rate, aes(k, error_rates_euc, error_rates_manh), axis = T) 

p + geom_line( aes(k, error_rates_euc, col = "euclidean")) + 
  geom_line( aes(k, error_rates_manh, col = "manhattan")) + 
  scale_x_discrete(limits = c(1:15)) + 
  scale_y_continuous(name="Error Rate", limits=c(.02, .075)) + 
  ggtitle("10-Fold CV Error Rates")


# 7 creating confusion matrices for the 3 'best' combinations of k/method 

#modifying cv_error_knn function to output predictions to input into 
#confusion matrix
cv_predict_knn = function(distance_matrix, labels, k) {
  
  #setting the distance from a point to itself to infinity for 
  # all the points, per Ben's OH. 
  for (ind in split_indexes) {
    distance_matrix[ind,ind] <- Inf
  }
  
  predictions = apply(distance_matrix, 2, function(col) {
    
    v = head(order(col), k)
    t = table(training_set[v,1])
    # print(t)
    result <- names(t)[which(t == max(t))]
    
    if(length(result) > 1)
      result <- sample(result, 1)
    factor(result)
  } )
  # browser()
  names(predictions) <- NULL
  predictions
  
}

# base on what we obsereved it looks like the best combinations are in k = 3-5 for 
# both methods explored (euclidean and manhattan). we exclued k= 1 case since it's most likely 
# to show a biased result by comparing a value to itself
# the 3 best combinations are [k=3, euclidean], [k = 3, manhattan*] , [k = 4, euclidean]


#cv predictions and confusion matrix for [k=3, euclidean]
confusion_euclid_k3 = cv_predict_knn(distance_matrix_euc, training_set[,1], 3)
confusion_matrix_1 <- table(training_set[,1], confusion_euclid_k3)
confusion_matrix_1

#cv predictions and confusion matrix for [k=3, manhattan]
confusion_manhattan_k3 = cv_predict_knn(distance_matrix_manh, training_set[,1], 3)
confusion_matrix_2 <- table(training_set[,1], confusion_manhattan_k3)
confusion_matrix_2

#cv predictions and confusion matrix for [k=4, euclidean]
confusion_euclid_k4 = cv_predict_knn(distance_matrix_euc, training_set[,1], 4)
confusion_matrix_3 <- table(training_set[,1], confusion_euclid_k4)
confusion_matrix_3



#Computing error rates for test set for K = 1,..., 15

#the approach we use here is implementing a hybrid function between
# predict_knn() and cv_error_knn(). Like in the latter, we compute
# the distance matrix beforehand and pass it as an argument into our
#function error_rate_knn(). And like a former, it takes our training and 
#test data, including labels for indexing and dimention control, but does
# not perform a cross validation.


#combining test and training set
mt_combined2 = as.matrix(rbind(training_set, test_set))

#computing distance matrix for 
test_set_distance_matrix_euc = compute_distances(mt_combined2[,-1], "euclidean")
test_set_distance_matrix_euc <- as.matrix(test_set_distance_matrix_euc)            #converting distnces into matrix
test_set_distance_matrix_manh = compute_distances(mt_combined2[,-1], "manhattan")
test_set_distance_matrix_manh <- as.matrix(test_set_distance_matrix_manh)          #converting distnces into matrix

#modifying pedict_knn function to return error rates instead of predictions
error_rate_knn = function(points, train, distance_matrix, labels, k, method) {
  
  n = nrow(train)
  m = nrow(points)
  
  #setting the distance from a point to itself to infinity for 
  # all the points, per Ben's OH. 
  for (ind in 1:(n+m)) {
    distance_matrix[ind,ind] <- Inf 
  }
  
  f = function(col) {
    v = head(order(col), k)
    t = table(labels[c(v)])

    result <- names(t)[which(t == max(t))]
    
    if(length(result) > 1)
      result <- sample(result, 1)
    result
  }
  
  predictions <- apply(distance_matrix, 2, f)
  names(predictions) <- NULL
  mean(predictions != labels)
}

test_error_rates_euc = c(1:15)

for (i in seq_along(k)) {
  
  test_error_rates_euc[i] = error_rate_knn(test_set[,-1], training_set[,-1],test_set_distance_matrix_euc, training_set[,1], i)  
  test_error_rates_euc[i]
}

test_error_rates_manh = c(1:15)

for (i in seq_along(k)) {
  
  test_error_rates_manh[i] = error_rate_knn(test_set[,-1], training_set[,-1],test_set_distance_matrix_manh, training_set[,1], i)  
  test_error_rates_manh[i]
}

#combining test set error rates for both methods into a dataframe for 
#easy plotting
test_set_error_rates <- data.frame(k, test_error_rates_euc, test_error_rates_manh)


#Plotting the test set error rates results for k = 1, ..., 15

p <- ggplot(test_set_error_rates, aes(k, test_error_rates_euc, test_error_rates_manh), axis = T) 

p + geom_line( aes(k, test_error_rates_euc, col = "euclidean")) + 
  geom_line( aes(k, test_error_rates_manh, col = "manhattan")) + 
  scale_x_discrete(limits = c(1:15)) + 
  scale_y_continuous(name="Error Rate", limits=c(.2,.4)) + 
  ggtitle("Test Set Error Rate")






 