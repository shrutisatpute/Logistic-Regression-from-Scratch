#Shruti Sachin Satpute
# 700701371
#Certificate of Authenticity: “I certify that the codes/answers of this assignment are entirely my own work.”

import pandas as pd
import numpy as np
from random import random

class Logistic_Regression:
    def __init__(self, train_file, test_file):
        train_file = train_file
        test_file = test_file
        self.logistic_train = pd.read_csv(train_file, header=None, delim_whitespace = True)
        self.logistic_test = pd.read_csv(test_file, header=None, delim_whitespace = True)

        # seperating all the columns from the train data except for the class variable.
        log_train = self.logistic_train.iloc[:,:-1]
        log_test = self.logistic_test.iloc[:,:-1]

        # Collecting the normalized data.
        log_train_norm = self.normalize(log_train)
        log_test_norm = self.normalize(log_test)

        # collecting the randomly generated theta
        log_train_theta = self.theta(len(log_train.iloc[0]))

        # Creating the hypothesis
        tanspose = np.dot(log_train_norm, log_train_theta)

        # hypothesis
        hypothesis = self.train_hypothesis(log_train_norm, log_train_theta)

        # creating class vector
        train_matrix = self.logistic_train.as_matrix()
        class_matrix = np.zeros((train_matrix.shape[0], 10))
        for i in range(10):
            class_matrix[:,i] = np.where(train_matrix[:,0]==i, 1,0)

        classwise_cost = list()
        #calculating cost for each class
        for i in range(10):
            classwise_cost.append(self.cost_function(hypothesis, class_matrix[:,i]))
        
        #optimization of thetas
        new_thetas = list()
        for j in range(len(classwise_cost)):
            new_thetas.append(self.optimization_function(log_train_norm,classwise_cost[j],hypothesis, log_train_theta, class_matrix[:,j] ))
            
    
        # seperating class variable.
        class_var_train = self.logistic_train.iloc[:,-1]
        class_var_test = self.logistic_test.iloc[:,-1] 
        correct_count = self.predition(log_test_norm, log_train_theta, class_var_test)
        print('\n\n', correct_count)
        percent = (correct_count/len(log_test_norm))*100
        print('\n\nAccuracy: ', percent,'%')

    # Normalization
    def normalize(self, filename):
        filename_mean = filename.mean()
        filename_sd = filename.std()
        filename_normalize = (filename - filename_mean)/filename_sd
        return filename_normalize

    # Generating the theta
    def theta(self, columns):
        thet_list = list()
        for i in range(columns):
            r = random()
            thet_list.append(r)
        return thet_list
    def train_hypothesis(self, log_normalized_train, thet_list):
        return self.sigmoid(np.dot(log_normalized_train, thet_list))
    def sigmoid(self, tanspose):
        return 1 / (1 + np.exp(-tanspose))

    #class_vector = np.zeros((data_full_matrix.shape[0],10))

    

    # cost function
    def cost_function(self, hypothesis, train_matrix):
        # for i in range(len(train_matrix[0])):
        final_cost = 0.0
        train_matrix = train_matrix.tolist()
        rows = len(hypothesis)
        for i in range(rows):
            final_cost += -(train_matrix[i]*np.log(hypothesis[i]) + (1-train_matrix[i])*np.log(1-hypothesis[i]))
        return final_cost/len(train_matrix)     

    def optimization_function(self, log_train_file, classwise_cost,hypothesis, thet_list, class_matrix):
        diff_list = self.difference(hypothesis, class_matrix)
        rows = len(hypothesis)
        columns = len(log_train_file.iloc[0])
        t = thet_list.copy()
        new_cost =0.0
        while new_cost > 0.002:
            for i in range(columns):
                for j in range(rows):
                    sum+= log_train_file.iloc[j,i]*diff_list
                diff = sum/rows
                t[i] = t[i] - (0.001*diff) # multiplying learning rate alpha with difference
            h = self.train_hypothesis(log_train_file, t)
            new_cost = self.cost_function(h, class_matrix)/rows
        
        return t

    def difference(self, hypothesis, class_matrix):
        return hypothesis-class_matrix


    def predition(self, logistic_test, thet_list, class_var_test):
        counter = 0
        for i in range(len(logistic_test)):
            prediction_list = list()
            for j in range(len(thet_list)):
                result = self.train_hypothesis(class_var_test[i], thet_list[j])
                prediction_list.append(result)
            real_value = class_var_test[i]
            outcome = prediction_list.index(max(prediction_list))

            if real_value == outcome:
                counter +=1
                print('Object ID: ',i,' || Predicted Class: ', outcome,' || True Class: ', real_value,' || Accuracy: ',1)
            else:
                print('Object ID: ',i,' || Predicted Class: ', outcome,' || True Class: ', real_value,' || Accuracy: ',0)
        return counter
    
logreg = Logistic_Regression(input("Train "), input("Test "))
    




                






