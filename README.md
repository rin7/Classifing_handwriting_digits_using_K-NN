Classifing handwriting digits 0-9 using K-NN

Description:
The U.S. Postal Service processes an average of 21.1 million pieces of mail per hour. Outbound mail must be
sorted according to the zip code of its destination. In the past, postal workers sorted mail by hand, which was
tedious and expensive. Over the last 40 years, USPS has switched to automated mail sorting. The sorting
machines use statistical classifiers to identify the individual digits in the zip code on each piece of mail.
We used the k-nearest neighbors algorithm for classifying handwritten digits in
zip codes. Zip codes only contain the numbers 0 through 9.
The k-nearest neighbors algorithm classifies an observation based on the class labels of the k nearest
observations in the training set (the ”neighbors”). The effectiveness of k-nn depends on the choice of k and
on how distance is measured between observations. Distance can be measured with real-world Euclidean
distance or with more exotic metrics such as Manhattan distance and Minkowski distance1.
The data set is split into a training set and a test set. Both have the same format. In the files, each line is
one observation (one digit). There are 257 entries on each line, separated by spaces. The first entry is the
class label for the digit (0–9) and the remaining 256 entries are the pixel values in the 16  16 grayscale
image of the digit. The pixel values are standardized to the interval [-1; 1]. There are 7291 observations in
the training file and 2007 observations in the test file.

contributors: 
  Arin Sadeghi
  Alexey Silin
  Luna Qiu
