# DataMining_Project
The dataset which we are using in this project is a huge dataset, so, while uploading the dataset to git, it showing error like upload files less than 25MB.
So, inorder to execute the given python file, we can follow the process mentioned below:
1. First, we have to download the dataset from the Kaggle platform with the link:
Link to download the creditcard dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Then, after extracting the downloaded dataset file, we have to give the path of this dataset file in our code file in the line:
#loading dataset here
fraud_data_frame = pd.read_csv(r"path/creditcard.csv")
So, in the above line, we have to replace the path with the actual local path that where we are storing our dataset.
3. Now, we can execute the python file to get the evaluation results.
