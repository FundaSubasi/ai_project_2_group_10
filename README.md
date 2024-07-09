# AI Project 2 Group 10
A data analysis class project for Columbia University's AI bootcamp.

## Reel Returns: Machine Learning Insights into Movie Profitability

<div align='center'>
    <img src='' height='300' title='AI Jobs'(image courtesy of Pexels) alt='Image of movie reels with popcorn'/>
</div>

## Project Team Members:
* Eric Alicea
* Kalvin Anglin
* Vadim Bychok
* Peta-Gaye McKenzie
* Odele Pax
* Funda Subasi

## Table of Contents

* [Abstract](#Abstract)
* [Data](#Data)
* [Methods](#Methods)
* [Limitations](#Limitations)
* [Conclusions](#Conclusions)
* [References/Footnotes](#References/Footnotes)

## Abstract

* Objective: To build a reliable predictive model to help industry stakeholders make an informed decision regarding movie investments and releases. This project aims to explore the relationship between consumer spending, inflation, unemployment and the performance of movies released in the US from 1981 - 2023 using machine learning techniques and analysis of key features such as, budget, genres,ratings, and economic indicators. 

* **Thesis/Hypothesis**: There is a correlation between the ratings and ROI of movies released in the US and the country's economic climate from 1981 - 2023. This is based on correlations between economic periods of bust and boom, and how movie ticket sales coincided with consumers’ spending habits, inflation, and unemployment. 
    * Considerations: Are movies more profitable in a depressed economy? Are movies recessionproof? Are people more inclined to spend money for movies during hard economic times? What kind of features have more weight in predicting a movie's success based on ratings, ROI and key economic indicators?


---

## Data

We reviewed datasets that provided the most opportunity for a thorough exploration of our key questions:

**Movies**
* Movies Dataset: 
https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates?select=TMDB_all_movies.csv

**Economic Indicator**

* Umemployment Dataset:
https://www.epi.org/data/#?subject=unemp

* Consumer Confidence Indicator:
https://www.kaggle.com/datasets/iqbalsyahakbar/cci-oecd

* Public Debt:
https://www.kaggle.com/datasets/pavankrishnanarne/us-public-debt-quarterly-data-1996-present

---

## Project Management
|  Phases | Details|
| :--- | :--- |
| Data Fetching  | Google searches, Kaggle, EPI Data Library, Census API search    |
| Software Version Control | Repository created on GitHub, GitHub Projects used to create and track tasks based on key questions, Git branch used to upload files to from local computer to remote repository, utilized "compare & pull requests" to compare branch changes before merging into the main branch correlation, comparison, summary statistics, sentiment analysis, and time series analysis   |
| EDA | Imported CSV files, created dataframes, utlized pandas and python functions to search, select and handle missing data movies and economoic dataset, identified keys features , experience levels, skills, and employment trends for further analysis    |
| Preprocessing  |  Utilized dictionarys, list, loops, column slicing, string manipulation, and train-test split to prevent data leakage and ensure high quality and structured data is visualized and fed into the models for linear regression predictions and classification   |
| Modeling & Tuning |  Utilized dictionarys, list, loops, column slicing, string manipulation, to ensure high quality and structured data is visualized and fed into the models for linear regression predictions and classification   |

---
## Methods 

* Linear Regression vs Logistic Regression 
  * Linear Regression is a method used in machine learning to predict a continuous outcome based on one or more input features. It works by finding the best-fit line that represents the relationship between the inputs and the output. This line is determined so that it minimizes the difference between the actual and predicted values. Essentially, Linear Regression helps in understanding and modeling how changes in the input variables affect the output variable, providing a straightforward way to make predictions.
  * Logistic Regression is a machine learning algorithm used for binary classification tasks, where the goal is to predict one of two possible outcomes. Unlike Linear Regression, which predicts continuous values, Logistic Regression predicts the probability that a given input belongs to a certain class. It works by applying a logistic function (also known as the sigmoid function) to the linear combination of input features. This function maps the output to a value between 0 and 1, which can be interpreted as a probability. Based on this probability, the final classification decision is made, typically by setting a threshold (e.g., 0.5) to determine the class. Logistic Regression is widely used because of its simplicity, efficiency, and effectiveness for binary classification problems.

* Classifier Models
  * K-Nearest Neighbor (KNN) is a straightforward algorithm used for classification and regression. It makes predictions by finding the k closest training examples to the new input and using their values. For classification, it assigns the most frequent class among the neighbors; for regression, it averages their values. KNN is easy to implement but can be slow with large datasets due to the need to compute distances for each prediction.
  * Random Forest is an ensemble learning method used for classification and regression tasks. It works by creating multiple decision trees during training and merging their results to improve accuracy and control overfitting. Each tree in the forest is built from a random subset of the training data and features, making the model more robust and less sensitive to noise. Predictions are made by averaging the results (regression) or taking a majority vote (classification) from all the trees. Random Forest is popular for its high accuracy, versatility, and ability to handle large datasets with many features.
  * AdaBoost is an ensemble learning method that boosts the performance of weak classifiers by combining them. It works iteratively, training weak classifiers and adjusting weights to focus on misclassified points. The final model is a weighted sum of these classifiers, improving accuracy and reducing bias and variance.
  * Decision Tree is a machine learning algorithm used for classification and regression. It splits data based on feature values to create a tree-like structure of decisions and outcomes. Each split is based on criteria like Gini impurity or information gain. The process continues until a stopping condition is met. Decision Trees are easy to interpret but can overfit without proper pruning.

---
## Results
* Linear Regression

  Selection of this model was based on relationship between the numerical response and one or more variables that explain the response. The numerical response or target variable we chose was ROI, however achieving a MSE = .0262 and R2 score = -.0086. While the MSE value is relatively small, R2 score is the statistical measure representing how well the model fits the data. A slightly negative R2 score suggests  our model is not fitting the data and performing worse than simply using the mean of the ROI as a prediction. The results of this model guide our selection of subsequent models better equipped for handling classification predictions and nonlinear patterns and trends.

* Logistic Regression
  
  This model was chosen to explore the classification of discrete variables. We featured engineered a target variable the contained difference classes of movie's success based on Raings, ROI, and certain economic indicators. As a statistical method for predicting binary outcomes, the Logistic Regression model achieved an accuracy score = .78 and precision score = .79 when it came to predicting the classification of target variable. Similar to Linear Regression, Logistic Regression is based on a linear relationship but different given it is between the the independent variable and the log-odds or probablity score between 0 and 1 for classification within the dependent variable. The scores for this model demonstrated it's ability to handle categorical outcomes effectively on large sample sizes. Larger dataset are recommended for Logistic regression models it provides better stability, reliability, and generalizability of the models estimates. A key drawback being this model's sensitivity to imbalanced classes in our dataset leading to biased predictions.

* K-Nearest Neighbor (KNN)
  
  In implementing the KNN to compare data points to determine the classfication of our target variable, we utilized a several variations of the model. First, we tested an untuned model which achieved an accuracy score of .65 and and a precision score of .62. Next, we tested the untuned model using PCA to reduce the dimensionality of the data, however the untuned model yielded an even lower accuracy score = .62 and a precision score = .59. To the tune the KNN classfier, we had the model loop through different k-values to find wich had the highest accuracy. The optimal k-value was 23, which yielded an similar accuracy score = .65 and slightly lower precision score =.60. Lastly, we hyperparameter tuned the KNN classifier using `GridSearchCV model` which yield k-value equal to 1 with the lowest accuracy score of .58. The lower scores of this model can attributed to the "Curse of Dimensionality" with causes the algorithm to perform poorly with high-dimensional data due to the distances between data points becoming less meaningful.

* Random Forest 
  
  The accuracy and precision scores of the Logistic Regression model indicate the Random Forest model should be tested due to it being robust to outliers and nonlinear data. Random Forest utilized a bagging technique by training multiple models independently and then combining their predictions into a final predictions. This model achieved an accuracy = .80 and precision = .83 illustrating it's success with large datasets and handling thousand of input variables by ranking importance in a natural way.

* AdaBoost

  We selected AdaBoost to test another ensemble learning model, but instead evaluate the boosting method. Boosting is technique where a strong learning model is created by increasing the weights of samples misclassified by multiple weak learner so the get more get more attention in the subsequent iterations. We tested this model, intially designed for binary classification on high dimensional dataset test is the "Sensitivity to Noisy Data" and achieved an accuracy = .37 and precision = .17. This scores were attributed to the models technique of increasing the weight of misclassification, while not accounting for those misclassifications being caused by noisy dataset.

* Decision Tree

  Testing out the Decision Tree model provided a basis for how a flowchart-like algorithm would perform on our dataset. The model used nodes to split the data into subsets based on the most significant features, and repeated this process until reaching a certain stopping criterion like maximum depth or minimum samples. We achieved an accuracy and precision score = .99,illustrating another models sensitivity to a dataset set with imbalanced classes. As a result this models scores were most likely attributable to the model's tendency to make predictions towards a dominant class.
___
## Future Considerations

* Gather more diverse economic data, and engage with an economist to interpret the data.
* Feature Selection and additional analysis to determine how the weights of the different features affect the ROI and could improve the model scores
* **Lingering Questions**
    * **Datasets:** Assumption "Revenue" feature in the TMDB Movies Dataset included streaming. Assumption "CCI, CPI, & Unemployment" are appropriate metrics for economic climate.


## Conclusion

* Our linear regression analysis of the movie industry between the years of 1981 and 2023, based on the relationship between a continuous dependent variable and key independent features in the dataset resulted in the conclusion that there was no clear correlation between a movie's ROI (the continuous dependent variable) and the following key independent features: consumer confidence index (CCI), consumer price index (CPI), and unemployment rate (UR). This was further supported by a Linear Regression R2 score = -.0086.

* Our predictive classification analysis of the movie industry during the same window of time, but based on the categorization of a movie's Success Conditions resulted in a Random Forest accuracy = .80 and precision = .83. The classes or conditions of success created by combining movie ratings, ROI, CCI, CPI, & UR provided an economic snapshot of a movie's performance. This snapshot provides valuable insight for investors and movie producer when determing the optimal time for a movie release.



## References/Footnotes
