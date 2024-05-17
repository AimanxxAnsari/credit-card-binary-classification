# credit-card-binary-classification

## Introduction
The credit card approval process plays a significant role in determining the financial well-being of individuals. The ability to predict the approval or the rejection of credit card applications is crucial for financial organizations to manage risk effectively and make decisions based on that. In this report, we are using various machine learning models to analyze the customer's data and automate the process of binary classification to predict the possibility of customers' attributes to get their applications approved or rejected.

## Analysis of the dataset
The credit card applicants' datasets for approval of the application on binary classification problem The dataset consists of 15 features, namely, ID, Gender, Car Owner, Property Owner, Number of childern, Annual Income, Type of Income, Education Level, Marital Status, Housing Type, Birthday Count, Employed Days, Type of Occupation, Family Members, Rejected Applications. The dataset consists of over 1500 observations and consists of various data types such as integer to object datatype.

Before diving into the Exploratory Data Analysis (EDA), we need to make sure the dataset is understandable and well-organized, preparing it is necessary before beginning the Exploratory Data Analysis (EDA). This includes fixing mistakes, dealing with missing data, eliminating anomalies, and carrying out any required adjustments. The primary tasks carried out during the preprocessing phase are as follows:

1. Handling Missing Values: Initially, we need to identify the missing values. For attributes - Annual Income and Age, we filled the null values with the mean of each attribute respectively.
2. Handling Duplicate Values: Redundant data in the dataset hinders maintaining data integrity. The duplicate data in the dataset has been removed to eliminate the possibility of inaccurate results.
3. Treating Anomalies: Anomaly detection and treating anomalies is essential for any dataset since it can skew the distribution heavily if not treated right.
4. Binary Encoding: For categorical data, we map each categorical value with its binary equivalent. For attributes, Property Owner, Car Owner, and Gender where the values are either Yes or No, or Male and Female. We have mapped each value as 0s and 1s.
5. Converting relevant values into meaningful values: Since the dataset consisted of negative values for birthday count (in days) and employed days (in days), the data has been converted to a more meaningful format by converting them into Age (in years) and Work Experience (in years).

![Alt text](imgs/corr.png)