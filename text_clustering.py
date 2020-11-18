# Problem 2: Part 1

# Reference: https://github.com/mayank408/TFIDF/blob/master/TFIDF.ipynb

# import all the packages
import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt
import re
import math

# read the csv file
# change the filepath to run in your system
data = pd.read_csv('P://Spring_2020//Data_Mining//HomeWork//Amazon_Reviews.csv')
review = data['Review']


# function to convert data to lower case
def convert_lower_case(data):
    return np.char.lower(data)


# function to remove punctuations
def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data


# function to remove apostrophe
def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


# function to preprocess data by calling other functions
def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data)
    data = remove_apostrophe(data)
    return str(data)


no_of_reviews = review.shape[0]

# append each review by spliting into strings
row = []
for i in range(0, no_of_reviews):
    row.append(preprocess(review[i]).strip().split(" "))

# combine all the words in row_combined
row_combined = []
for i in range(0, no_of_reviews):
    row_combined.extend(row[i])

# set of all the words
wordSet = set(row_combined)

# from the wordSet, pick each word as a key and assign 0 as value
wordDictionary = []
for i in range(0, no_of_reviews):
    wordDictionary.append(dict.fromkeys(wordSet, 0))

# run through each review and create dictionary
for i in range(0, no_of_reviews):
    for word in row[i]:
        wordDictionary[i][word] += 1


# function to calculate TF
def TF(wordDictionary, row):
    tfDict = {}
    rowCount = len(row)
    for word, count in wordDictionary.items():
        tfDict[word] = count / float(rowCount)
    return tfDict


# append wordDict for each review
tf = []
for i in range(0, no_of_reviews):
    tf.append(TF(wordDictionary[i], row[i]))


# function to calculate IDF
def IDF(lists):
    N = len(lists)

    idfDict = dict.fromkeys(lists[0].keys(), 0)
    for doc in lists:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))

    return idfDict


# append wordDict for all the words in a list
combined_word_dict = []
for i in range(0, no_of_reviews):
    combined_word_dict.append(wordDictionary[i])

idf = IDF(combined_word_dict)


# function to calculate TFIDF
def TFIDF(tf, idf):
    tfidf = {}
    for word, val in tf.items():
        tfidf[word] = val * idf[word]
    return tfidf


tfidf = []
for i in range(0, no_of_reviews):
    tfidf.append(TFIDF(tf[i], idf))

# append tfidf weight for each review
combined_tfidf = []
for i in range(0, no_of_reviews):
    combined_tfidf.append(tfidf[i])

# final nxm matrix
matrix = pd.DataFrame(combined_tfidf)

print('Output: Part 1')
print(matrix)

# plot 2D image
plt.title('TF-IDF Weight Matrix Heatmap')
plt.imshow(matrix, interpolation='nearest')
plt.axis('tight')
plt.show()

# Problem 2: Part 2

# import all the packages
import pandas as pd
import numpy as np
import math

# read the csv file
# change the filepath to run in your system
data = pd.read_csv('P://Spring_2020//Data_Mining//HomeWork//Amazon_Reviews.csv')
review = data['Review']


# function to convert data to lower case
def convert_lower_case(data):
    return np.char.lower(data)


# function to remove punctuations
def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data


# function to remove apostrophe
def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


# function to preprocess data by calling other functions
def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data)
    data = remove_apostrophe(data)
    return str(data)


no_of_reviews = review.shape[0]

# list of positive and negative words
positive = ["great", "awesome", "excellent", "loved", "good"]
negative = ["beware", "bad", "disappointed", "pathetic", "ridiculous"]

# append each review by spliting into strings
row = []
for i in range(0, no_of_reviews):
    row.append(preprocess(review[i]).strip().split(" "))

# set of all positive and negative words
wordSet = set(positive).union(set(negative))

# from the wordSet, pick each word as a key and assign 0 as value
wordDictionary = []
for i in range(0, no_of_reviews):
    wordDictionary.append(dict.fromkeys(wordSet, 0))

# run through each review and create dictionary
for i in range(0, no_of_reviews):
    for word in wordSet:
        if word in row[i]:
            wordDictionary[i][word] += 1


# function to calculate TF
def TF(wordDictionary, row):
    tfDict = {}
    rowCount = len(row)
    for word, count in wordDictionary.items():
        tfDict[word] = count / float(rowCount)
    return tfDict


# append wordDict for each review
tf = []
for i in range(0, no_of_reviews):
    tf.append(TF(wordDictionary[i], row[i]))


# function to calculate IDF
def IDF(lists):
    N = len(lists)

    idfDict = dict.fromkeys(lists[0].keys(), 0)
    for doc in lists:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))

    return idfDict


# append wordDict for all the words in a list
combined_word_dict = []
for i in range(0, no_of_reviews):
    combined_word_dict.append(wordDictionary[i])

idf = IDF(combined_word_dict)


# function to calculate TFIDF
def TFIDF(tf, idf):
    tfidf = {}
    for word, val in tf.items():
        tfidf[word] = val * idf[word]
    return tfidf


tfidf = []
for i in range(0, no_of_reviews):
    tfidf.append(TFIDF(tf[i], idf))

# append tfidf weight for each review
combined_tfidf = []
for i in range(0, no_of_reviews):
    combined_tfidf.append(tfidf[i])

# final n x 10 matrix of tf-tdf
tf_idf_matrix = pd.DataFrame(combined_tfidf)

# final n x 10 count matrix
count_matrix = pd.DataFrame(wordDictionary)

print('\nOutput: Part 2')
print('TF-IDF MATRIX')
print(tf_idf_matrix)

print('COUNT MATRIX')
print(count_matrix)

# Problem 2: Part 3

# import all the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

# read the csv file
# change the filepath to run in your system
data = pd.read_csv('P://Spring_2020//Data_Mining//HomeWork//Amazon_Reviews.csv')
review = data['Review']

# function to convert data to lower case
def convert_lower_case(data):
    return np.char.lower(data)

# function to remove punctuations
def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

# function to remove apostrophe
def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

# function to preprocess data by calling other functions
def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data)
    data = remove_apostrophe(data)
    return str(data)

no_of_reviews = review.shape[0]

# list of positive and negative words
positive = ["great", "awesome", "excellent", "loved", "good"]
negative = ["beware", "bad", "disappointed", "pathetic", "ridiculous"]

# append each review by spliting into strings
row = []
for i in range(0, no_of_reviews):
    row.append(preprocess(review[i]).strip().split(" "))

# set of positive and negative words
wordSetP = set(positive)
wordSetN = set(negative)

# from the wordSetP, pick each word as a key and assign 0 as value
wordDictPositive = []
for i in range(0, no_of_reviews):
    wordDictPositive.append(dict.fromkeys(wordSetP, 0))

# from the wordSetN, pick each word as a key and assign 0 as value
wordDictNegative = []
for i in range(0, no_of_reviews):
    wordDictNegative.append(dict.fromkeys(wordSetN, 0))

# run through each review and create dictionary of positive words
for i in range(0, no_of_reviews):
    for word in wordSetP:
        if word in row[i]:
            wordDictPositive[i][word] += 1

# calculate sum of frequencies of all the positive words in each review
sumP = []
for i in range(no_of_reviews):
    sumP.clear()
    for word in wordDictPositive:
        s = 0
        for i in word:
            s = s + word[i]
        sumP.append(s)

# run through each review and create dictionary of negative words
for i in range(0, no_of_reviews):
    for word in wordSetN:
        if word in row[i]:
            wordDictNegative[i][word] += 1

# calculate sum of frequencies of all the negative words in each review
sumN = []
for i in range(no_of_reviews):
    sumN.clear()
    for word in wordDictNegative:
        s = 0
        for i in word:
            s = s + word[i]
        sumN.append(s)

# final n x 2 matrix
column_name = ['positive', 'negative']

pos_matrix = pd.DataFrame([sumP, sumN])

pos_matrix.index = ['Positive', 'Negative']

print('\nOutput: Part 3')
print('FREQUENCY MATRIX')
print(np.transpose(pos_matrix))

# visualising using kmeans clustering
# zip both axis together to form 2D array
dataSet = np.array(list(zip(sumP, sumN)))

# define mykmeans function
def mykmeans(X, k, c):
    # step 1: randomly select center
    # call initialize function to randomly select initial center from the given centers
    initial_center = initialize(c, k)
    cluster = X.shape[0]
    iteration = 0
    while True:
        # iterate until iteration < 10000
        for iteration in range(10000):
            iteration += 1
            # step 2: calculate distance between initial center and points
            distances = pairwise_distances(X, initial_center)
            # find minimum centroid
            cluster = closest_centroid(distances)
            # store old centroid
            centroid_old = initial_center.copy()
            # step 3: calculate new centers from mean of points and update new centers to initial_center
            initial_center = average(X, cluster, k)
            # break if there is no difference in old and new centroids
            if np.all(centroid_old == initial_center):
                break
            # break if distance between old and new centroids is less than 0.001
            if distance(initial_center, centroid_old, 1).any() <= 0.001:
                break
        return initial_center, cluster, iteration

# randomly select initial centeroid
def initialize(c, k):
    centroids = c.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

# calculate euclidean distance
def distance(a, b, axis = 1):
    return np.linalg.norm(a - b, axis)

# find minimum distance
def closest_centroid(distances):
    return np.argmin(distances, axis=1)

# calculate new centers from mean of points
def average(X, cluster, k):
    return np.array([X[cluster == i].mean(0) for i in range(k)])


# Scatter plot for k = 2,3,4 and centers of each cluster in the range [0, 2]
# Uncomment each cluster to run the code

# print('k = 2')
# k = 2
# c1 = np.array([(0, 0)])
# c2 = np.array([(1, 1)])
# c = np.concatenate((c1, c2), axis = 0)

# center, cluster, iteration = mykmeans(dataSet, k, c)
# print('Centers: ')
# print(center)
# print('Iterations: ')
# print(iteration)

# plt.xlabel('Positive')
# plt.ylabel('Negative')

# plt.scatter(dataSet[:, 0], dataSet[:, 1], c=cluster, s=5)
# plt.scatter(center[:, 0], center[:, 1], c='red', marker='*', s=50, alpha=0.5)
# plt.xlim(-2,10)
# plt.ylim(-6,6)
# plt.axis('equal')
# plt.show()

# print('k = 3')
# k = 3
# c1 = np.array([(0, 0)])
# c2 = np.array([(1, 1)])
# c3 = np.array([(1, 2)])
# c = np.concatenate((c1, c2, c3), axis = 0)

# center, cluster, iteration = mykmeans(dataSet, k, c)
# print('Centers: ')
# print(center)
# print('Iterations: ')
# print(iteration)

# plt.xlabel('Positive')
# plt.ylabel('Negative')

# plt.scatter(dataSet[:, 0], dataSet[:, 1], c=cluster, s=5)
# plt.scatter(center[:, 0], center[:, 1], c='red', marker='*', s=50, alpha=0.5)
# plt.xlim(-2,10)
# plt.ylim(-6,6)
# plt.axis('equal')
# plt.show()

print('k = 4')
k = 4
c1 = np.array([(1, 1)])
c2 = np.array([(1, 0)])
c3 = np.array([(0, 1)])
c4 = np.array([(1, 2)])

c = np.concatenate((c1, c2, c3, c4), axis=0)
center, cluster, iteration = mykmeans(dataSet, k, c)
print('Centers: ')
print(center)
print('Iterations: ')
print(iteration)

plt.xlabel('Positive')
plt.ylabel('Negative')
plt.scatter(dataSet[:, 0], dataSet[:, 1], c=cluster, s=5)
plt.scatter(center[:, 0], center[:, 1], c='red', marker='*', s=50, alpha=0.5)
plt.xlim(-2,10)
plt.ylim(-6,6)
plt.axis('equal')
plt.show()
