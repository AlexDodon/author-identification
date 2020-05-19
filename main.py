from dataManipulation import *
import tensorflow as tf
import numpy as np
from sklearn import svm
from sklearn import datasets
import math
from Autoencoder import *

f = open("./trainingDatasetTensor2", 'rb')
trainingDatasetTensor = pickle.load(f)
f.close()
f = open("./testingDatasetTensor2", 'rb')
testingDatasetTensor = pickle.load(f)
f.close()

input_dim = len(trainingDatasetTensor[0])
int_dim1 = input_dim / 2
int_dim2 = int_dim1
np.random.seed(1)
tf.random.set_seed(1)
batch_size = 4
epochs = 50
learning_rate = 1e-2

ae1 = Autoencoder(
  intermediate_dim=int_dim1, 
  original_dim=input_dim
)
opt = tf.optimizers.Adam(learning_rate=learning_rate)

training_dataset = tf.data.Dataset.from_tensor_slices(trainingDatasetTensor)
training_dataset = training_dataset.batch(batch_size)
training_dataset = training_dataset.prefetch(batch_size * 4)
testing_dataset = tf.data.Dataset.from_tensor_slices(testingDatasetTensor)
testing_dataset = testing_dataset.batch(batch_size)
testing_dataset = testing_dataset.prefetch(batch_size * 8)

print("Autoencoder 1\n")
for epoch in range(epochs):
  for step, batch_features in enumerate(training_dataset):
    train(loss, ae1, opt, batch_features)
    loss_values = loss(ae1, batch_features)
    print(" epoch {} step {} loss {}".format(epoch, step, loss_values))

ae2 = Autoencoder(
  intermediate_dim=int_dim2, 
  original_dim=int_dim1
)

firstCodeTensor = [ae1.encoder(x) for y in training_dataset for x in y]
training_dataset2 = tf.data.Dataset.from_tensor_slices(firstCodeTensor)
training_dataset2 = training_dataset2.batch(batch_size)
training_dataset2 = training_dataset2.prefetch(batch_size * 8)

print("Autoencoder 2")
for epoch in range(epochs):
  for step, batch_features in enumerate(training_dataset2):
    train(loss, ae2, opt, batch_features)
    loss_values = loss(ae2, batch_features)
    print(" epoch {} step {} loss {}".format(epoch, step, loss_values))

svm_training_dataset = [ae2.encoder(x).numpy() for x in firstCodeTensor ]
svm_testing_dataset = [ae2.encoder(ae1.encoder(x)).numpy() for y in testing_dataset for x in y ]

clf = svm.LinearSVC()
X_train = formatForSVM(svm_training_dataset)
X_test = formatForSVM(svm_testing_dataset)
Y_train =[i for i in range(len(svm_training_dataset) )]
Y_train[0] = 0 
Y_test = Y_train

estimation = clf.fit(X_train, Y_train)

dec = clf.decision_function(X_train)

predictions = estimation.predict(X_test)

print("\nPredictions:")
print(predictions)
print("\nActual labels:")
print(Y_test)

correct = 0
for i in range(len(svm_testing_dataset)):
  if predictions[i] == Y_test[i]:
    correct += 1

accuracy = correct / len(Y_test)
print("\nAccuracy: {}\n".format(accuracy))
