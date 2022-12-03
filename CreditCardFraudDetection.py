import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

#loading dataset here
fraud_data_frame = pd.read_csv(r"enter_path_to_dataset_here\creditcard.csv")
#Showing number of samples(rows) and features(columns) (rows, columns)
print("Dataset Shape:\n(Samples, features)\n", fraud_data_frame.shape)
#Showing all the feature names i.e., column names
print("The names of the features are:\n", fraud_data_frame.columns)

print(fraud_data_frame.head(5))
print(fraud_data_frame.tail(5))

#Showing all the unique values in the target variable (Class)
print("The unique values in the target variable 'Class' are:\n", fraud_data_frame['Class'].unique())

#Showing number of samples in both the target values ('1' for fraud transactions & '0' for non-fraud transactions)
print("The number of samples in each target value is:\n", fraud_data_frame['Class'].value_counts())

#Data preprocessing to improve the quality of the data
#Removing irrelevant features
fraud_data_frame = fraud_data_frame.drop(['Time'], axis = 1)
print("The names of the features after removing the Time column are:\n", fraud_data_frame.columns)

#Checking datatypes and non-null values of all the features
print("Dataset info:\n", fraud_data_frame.info())

#Data Transformation
print("Sample values of the Amount column in the dataset:\n", fraud_data_frame['Amount'][0:7])
fraud_data_frame['normalized_amount'] = StandardScaler().fit_transform(fraud_data_frame['Amount'].values.reshape(-1, 1))
fraud_data_frame = fraud_data_frame.drop(['Amount'], axis = 1)
print("Sample values of the new Amount column 'normalized_column' after applying the StandardScaler are:\n", fraud_data_frame['normalized_amount'][0:7])

#Splitting the dataset into independent columns as 'X' and the dependent column(here Class column) as 'Y'
X = fraud_data_frame.drop(['Class'], axis = 1)
Y = fraud_data_frame['Class']
 
#Splitting the dataset into train and test dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
print("X_train Shape:\n", X_train.shape)
print("Y_train Shape:\n", Y_train.shape)
print("X_test Shape:\n", X_test.shape)
print("Y_test Shape:\n", Y_test.shape)

#Developing the model
#Building the model with decision tree
def decision_tree_classification(X_train, Y_train, X_test, Y_test):
    print("DecisionTree")
    #initializing object for the DecisionTreeClassifier class
    decision_tree_classifier = DecisionTreeClassifier()
    #Training model by using the fit method
    print("Model training starts...")
    decision_tree_classifier.fit(X_train, Y_train.values.ravel())
    print("Model training completed!")
    accuracy_score = decision_tree_classifier.score(X_test, Y_test)
    print("The accuracy score of model on the test dataset is: ", accuracy_score)

    #Predicting the result using the test dataset
    Y_prediction = decision_tree_classifier.predict(X_test)
    
    #confusion matrix
    print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_prediction))
    #classification report for f1-score
    print("Classification report:\n", classification_report(Y_test, Y_prediction))
    #Area under roc curve
    print("AROC score:\n", roc_auc_score(Y_test, Y_prediction))

#Calling the decision_tree_classification method to train and evaluate the model
decision_tree_classification(X_train, Y_train, X_test, Y_test)

#Building the model with random forest algorithm
def random_forest_classifier(X_train, Y_train, X_test, Y_test):
    print("RandomForest")
    #initializing a object for DecisionTreeClassifier class
    random_forest_classifier = RandomForestClassifier(n_estimators = 50)
    #train model by using the fit method
    print("Model training starts...")
    random_forest_classifier.fit(X_train, Y_train.values.ravel())
    print("Model training completed!")
    accuracy_score = random_forest_classifier.score(X_test, Y_test)
    print("The accuracy of model on the test dataset is: ", accuracy_score)
    #predicting the result using the test dataset
    Y_prediction = random_forest_classifier.predict(X_test)
    #confusion matrix
    print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_prediction))
    #classification report for f1-score
    print("Classification Report:\n", classification_report(Y_test, Y_prediction))
    #Area under roc curve
    print("AROC score:\n", roc_auc_score(Y_test, Y_prediction))
#Calling the random_forest_classifier method to train and evaluate the model
random_forest_classifier(X_train, Y_train, X_test, Y_test)

#Building the model with Support Vector Machines (SVM) algorithm
def svm_classifier(X_train, Y_train, X_test, Y_test):
    print("SVM")
    #initializing a object for svm classifier class
    svm_classifier = svm.SVC()
    #training the model by using the fit method
    print("SVM model training starts...")
    svm_classifier.fit(X_train, Y_train.values.ravel())
    print("SVM model training completed!")
    accuracy_score = svm_classifier.score(X_test, Y_test)
    print("The accuracy of svm model on the test dataset is: ", accuracy_score)
    #predicting the result using the test dataset
    Y_prediction = svm_classifier.predict(X_test)
    #confusion matrix
    print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_prediction))
    #classification report for f1-score
    print("Classification Report:\n", classification_report(Y_test, Y_prediction))
    #Area under roc curve
    print("AROC score:\n", roc_auc_score(Y_test, Y_prediction))
#Calling the svm_classifier method to train and evaluate the model
svm_classifier(X_train, Y_train, X_test, Y_test)

#Building the model with K-nearest neighbors algorithm
def knn_classifier(X_train, Y_train, X_test, Y_test):
    print("KNN")
    #initializing a object for knn classifier class
    knn_classifier = KNeighborsClassifier(n_neighbors=7)
    #training the model by using the fit method
    print("KNN model training starts...")
    knn_classifier.fit(X_train, Y_train.values.ravel())
    print("KNN model training completed!")
    accuracy_score = knn_classifier.score(X_test, Y_test)
    print("The accuracy of knn model on the test dataset is: ", accuracy_score)
    #predicting the result using the test dataset
    Y_prediction = knn_classifier.predict(X_test)
    #confusion matrix
    print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_prediction))
    #classification report for f1-score
    print("Classification Report:\n", classification_report(Y_test, Y_prediction))
    #Area under roc curve
    print("AROC score:\n", roc_auc_score(Y_test, Y_prediction))
#Calling the knn_classifier method to train and evaluate the model
knn_classifier(X_train, Y_train, X_test, Y_test)

#Building the model with Naive bayes algorithm
def naive_bayes_classifier(X_train, Y_train, X_test, Y_test):
    print("Naive Bayes")
    #initializing a object for Naive Bayes classifier class
    naive_bayes_classifier = GaussianNB()
    #training the model by using the fit method
    print("Naive bayes model training starts...")
    naive_bayes_classifier.fit(X_train, Y_train.values.ravel())
    print("Naive bayes model training completed!")
    accuracy_score = naive_bayes_classifier.score(X_test, Y_test)
    print("The accuracy of naive bayes model on the test dataset is: ", accuracy_score)
    #predicting the result using the test dataset
    Y_prediction = naive_bayes_classifier.predict(X_test)
    #confusion matrix
    print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_prediction))
    #classification report for f1-score
    print("Classification Report:\n", classification_report(Y_test, Y_prediction))
    #Area under roc curve
    print("AROC score:\n", roc_auc_score(Y_test, Y_prediction))
#Calling the naive_bayes_classifier method to train and evaluate the model
naive_bayes_classifier(X_train, Y_train, X_test, Y_test)

#Building the model with Logistic Regression algorithm
def logistic_regression_classifier(X_train, Y_train, X_test, Y_test):
    print("Logistic Regression")
    #initializing a object for  logistic regression classifier class
    logistic_regression_classifier = LogisticRegression()
    #training the model by using the fit method
    print("Logistic Regression model training starts...")
    logistic_regression_classifier.fit(X_train, Y_train.values.ravel())
    print("Logistic Regression model training completed!")
    accuracy_score = logistic_regression_classifier.score(X_test, Y_test)
    print("The accuracy of logistic regression model on the test dataset is: ", accuracy_score)
    #predicting the result using the test dataset
    Y_prediction = logistic_regression_classifier.predict(X_test)
    #confusion matrix
    print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_prediction))
    #classification report for f1-score
    print("Classification Report:\n", classification_report(Y_test, Y_prediction))
    #Area under roc curve
    print("AROC score:\n", roc_auc_score(Y_test, Y_prediction))
#Calling the logistic_regression_classifier method to train and evaluate the model
logistic_regression_classifier(X_train, Y_train, X_test, Y_test)

#Checking the number of samples for each target value
print("The number of samples under the each target value are:\n", fraud_data_frame['Class'].value_counts())

#Target class distribution
class_value = fraud_data_frame['Class'].value_counts()
print("The number of samples under each class is\n", class_value)
fraud_value = class_value[1]
non_fraud_value = class_value[0]
print("Total fraudulent numbers: ", fraud_value)
print("Total non fraudulent numbers: ", non_fraud_value)

#UnderSampling techniques
#Making both the target samples to the same level
fraud_indices = np.array(fraud_data_frame[fraud_data_frame['Class'] == 1].index)
non_fraud_indices = fraud_data_frame[fraud_data_frame['Class'] == 0].index
#Taking random samples from non fraudulent that are equal to fraudulent samples
random_non_fraud_indices = np.random.choice(non_fraud_indices, fraud_value, replace = False)
random_non_fraud_indices = np.array(random_non_fraud_indices)

#Concatenate the both indices of fraud and non fraud
under_sample_indices = np.concatenate([fraud_indices, random_non_fraud_indices])

#Extracting all the features from whole data for under sample indices only
under_sample_data = fraud_data_frame.iloc[under_sample_indices, :]

#Now, we have to divide the under sampling data to all features and target
X_under_sample_data = under_sample_data.drop(['Class'], axis = 1)
Y_under_sample_data = under_sample_data[['Class']]

#Now, we have to split the dataset to train and test datasets like before
X_train_sample, X_test_sample, Y_train_sample, Y_test_sample = train_test_split(X_under_sample_data, Y_under_sample_data, test_size = 0.2, random_state = 0)

#Calling DecisionTree method after applying the sampling techniques
print("DecisionTree Result After Applying Sampling Techniques:")
decision_tree_classification(X_train_sample, Y_train_sample, X_test_sample, Y_test_sample)

#Calling RandomForest method after applying the sampling techniques
print("RandomForest Result After Applying Sampling Techniques:")
random_forest_classifier(X_train_sample, Y_train_sample, X_test_sample, Y_test_sample)

#Calling SVM method after applying the sampling techniques
print("SVM Result After Applying Sampling Techniques:")
svm_classifier(X_train_sample, Y_train_sample, X_test_sample, Y_test_sample)

##Calling KNN method after applying the sampling techniques
print("KNN Result After Applying Sampling Techniques:")
knn_classifier(X_train_sample, Y_train_sample, X_test_sample, Y_test_sample)

##Calling Naive Bayes method after applying the sampling techniques
print("Naive Bayes Result After Applying Sampling Techniques:")
naive_bayes_classifier(X_train_sample, Y_train_sample, X_test_sample, Y_test_sample)

##Calling Logistic Regression method after applying the sampling techniques
print("Logistic Regression Result After Applying Sampling Techniques:")
logistic_regression_classifier(X_train_sample, Y_train_sample, X_test_sample, Y_test_sample)

#Analysis of all the machine learning algorithms used so far
#Before applying the sampling techniques
f1_scores_class_0 = [100, 100, 100, 100, 99, 100]
f1_scores_class_1 = [80, 85, 77, 82, 12, 73]
models = ['DecisionTree', 'RandomForest', 'SVM', 'KNN', 'NaiveBayes', 'LogisticRegression']
X_axis = np.arange(len(models))
Y_axis = np.arange(len(f1_scores_class_0))
plt.bar(X_axis - 0.2, f1_scores_class_0, 0.4, label = "f1-score-class-0")
plt.bar(X_axis + 0.2, f1_scores_class_1, 0.4, label = "f1-score-class-1")
plt.xticks(X_axis, models)
plt.xlabel("Data Mining Models")
plt.ylabel("f1-scores")
plt.title("F1 scores for all the Data Mining techniques used before applying sampling")
plt.legend()
plt.show()

#After applying the under-sampling techniques
f1_scores_class_0 = [92, 96, 96, 98, 93, 96]
f1_scores_class_1 = [91, 96, 95, 97, 91, 95]
models = ['DecisionTree', 'RandomForest', 'SVM', 'KNN', 'NaiveBayes', 'LogisticRegression']
X_axis = np.arange(len(models))
Y_axis = np.arange(len(f1_scores_class_0))
plt.bar(X_axis - 0.2, f1_scores_class_0, 0.4, label = "f1-score-class-0")
plt.bar(X_axis + 0.2, f1_scores_class_1, 0.4, label = "f1-score-class-1")
plt.xticks(X_axis, models)
plt.xlabel("Data Mining Models")
plt.ylabel("f1-scores")
plt.title("F1 scores for all the Data Mining techniques used after applying under sampling")
plt.legend()
plt.show()

#Accuracy before and after applying sampling techniques
Accuracy_before = [99.93, 99.95, 99.93, 99.94, 97.84, 99.92]
Accuracy_after =[91.88, 95.94, 95.43, 97.46, 91.88, 95.43]
models = ['DecisionTree', 'RandomForest', 'SVM', 'KNN', 'NaiveBayes', 'LogisticRegression']
X_axis = np.arange(len(models))
Y_axis = np.arange(len(Accuracy_before))
plt.bar(X_axis - 0.2, Accuracy_before, 0.4, label = "Accuracy_before")
plt.bar(X_axis + 0.2, Accuracy_after, 0.4, label = "Accuracy_after")
plt.xticks(X_axis, models)
plt.xlabel("Data Mining Models")
plt.ylabel("Accuracy")
plt.title("Accuracy for all the Data Mining techniques used")
plt.legend()
plt.show()

#AROC Scores before and after applying sampling techniques
AROC_Scores_Before = [88.42, 88.43, 82.65, 86.39, 90.77, 80.95]
AROC_Scores_After =[91.99, 95.84, 95.13, 97.33, 91.44, 95.37]
models = ['DecisionTree', 'RandomForest', 'SVM', 'KNN', 'NaiveBayes', 'LogisticRegression']
X_axis = np.arange(len(models))
Y_axis = np.arange(len(AROC_Scores_Before))
plt.bar(X_axis - 0.2, AROC_Scores_Before, 0.4, label = "AROC_Scores_Before")
plt.bar(X_axis + 0.2, AROC_Scores_After, 0.4, label = "AROC_Scores_After")
plt.xticks(X_axis, models)
plt.xlabel("Data Mining Models")
plt.ylabel("AROC_Scores")
plt.title("AROC_Scores_After for all the Data Mining techniques used")
plt.legend()
plt.show()
