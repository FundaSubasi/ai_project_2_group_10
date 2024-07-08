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

## Abstract - Eric

### Project Details

* Objective: 

* **Thesis/Hypothesis**: Are movies more profitable in a depressed economy? Are movies recessionproof?

Are people more inclined to spend money for movies during hard economic times.


---

## Data

### Datasets

We reviewed datasets that provided the most opportunity for a thorough exploration of our key questions:

**Movies**
* Movies Dataset: 
https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates?select=TMDB_all_movies.csv

**Cost of Living**

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

## Methods - Vadim

* Linear Regression

* Logistic Regression

* K-Nearest Neighbor (KNN)

* Random Forest

## Results - Kalvin

### Findings
* Linear Regression: Selection of this model was based on relationship between the numerical response and one or more variables that explain the response. The numerical response or target variable we chose was ROI, however achieving a MSE = .0262 and R2 score = -.0086. While the MSE value is relatively small, R2 score is the statistical measure representing how well the model fits the data. A slightly negative R2 score suggests model our model is not fitting the data and performing worse than simply using the mean of the ROI as a prediction. The results of this model guide of selection subsequent models better equipped for handling nonlinear patterns and trends.

* K-Nearest Neighbor (KNN): In implementing the KNN to compare data points to determine the classfication of our target variable, we utilized a several variations of the model. First, we tested an untuned model which achieved an accuracy score of .65 and and a precision score of .62. Next, we tested the untuned model using PCA to reduce the dimensionality of the data, however the untuned model yielded an even lower accuracy score = .62 and a precision score = .59. To the tune the KNN classfier, we had the model loop 
through different k-values to find wich had the highest accuracy. The optimal k-value was 23, which yielded an similar accuracy score = .65 and slightly lower precision score =.60. Lastly, we hyperparameter tuned the KNN classifier using `GridSearchCV model` which yield k-value equal to 1 with the lowest accuracy score of .578

* Logistic Regression: This model was chosen to explore discrete variables instead of nominal variable like ROI. We featured engineered a target variable the contained difference classes of movie's success based on Raings, ROI, and certain economic indicators. As a statistical method for predicting binary outcomes, the Logistic Regression model achieved an accuracy score = .78 and precision score = .79 when it came to predicting the classification of target variable.

* Random Forest: The accuracy and precision scores of the Logistic Regression model indicate the Random Forest model should be tested due to it being robust to outliers and nonlinear data. This model achieved an accuracy = .80 and precision = .84 illustrating it's success with large datasets and handling thousand of input variables by ranking importance in a natural way.

___
## Future Considerations - Team

### Lingering Questions

**Datasets:** Assumption "Revenue" feature in the TMDB Movies Dataset included streaming. Assumption "CCI, CPI, & Unemployment" are appropriate metrics for economic climate.

Consolidate more data / consult with a economist

## Conclusion - Team

Based on annual analysis between the years of 1981 and 2023, the results indicate no clear relationshp between a movie's ROI and the economy's consumer confidence index, price index, and unemployment based on annual metrics.

We were hoping data would show that in bad economic times people are more likely to spend money to see on movies as a form of escapism. 

Target variable (reel_economy) is a concatenation of three different classes: critical success(Ratings ), financial success (ROI), and economic indicators(CCI,CPI,& UR)

|  Ratings | Critical Success|
| :--- | :--- |
| 0 - 2.4  | Panned    |
| 2.5 - 4.9 | Alright   |
| 5 - 7.4 | Well Liked   |
| 7.5 - 10 | Success   |

|  ROI (%) | Financial Success|
| :--- | :--- |
|  < 0  | Failure    |
| <= 50| Modest Returns  |
| <= 100 | Moderate Returns   |
| <= 500 | Excellent Returns |
| > 500 | Extraordinary Returns   |

|  CCI, CPI, UR | Economic Success|
| :--- | :--- |
|    | Comfortable to Good    |
| | Lean to Bad  |


Our models can provide the following business insights for a movie producer:
Predicting the ROI and determining the optimal time for movie releases (Economic climate is a kep part of the timing).



## References/Footnotes