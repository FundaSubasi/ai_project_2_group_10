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
* Linear Regression

* Logistic Regression

* K-Nearest Neighbor (KNN)

* Random Forest

### Lingering Questions
___
## Limitations - Team

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