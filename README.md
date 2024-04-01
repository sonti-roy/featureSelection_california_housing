![alt text](https://github.com/sonti-roy/california_housing/blob/main/plots/logo.png)


# Feature Selection using supervised and unsupervised method and model development on california housing dataset


## Implementation Details

- Dataset: California Housing Dataset (view below for more details)
- Feature selection methods:
    - Supervised feature selection
        - mutual info regression
        - f_regression
        - Pearson Correlation
        - Recurive Feature Elimination (RFE) with Lasso
        - Sequential Feature Selection with RedgeCV
     - Unsupervised feature selection
         - Principal component analysis
- Model evaluated: 
  - [Linear Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
  - [KNeighborsRegression](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
  - [SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
  - [BayesianRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
  - [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
  - [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- Input: 8 features - Median Houshold income, House Area, ...
- Output: House Price

## Dataset Details

This dataset was obtained from the StatLib repository ([Link](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html))

This dataset was derived from the 1990 U.S. census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

A household is a group of people residing within a home. Since the average number of rooms and bedrooms in this dataset are provided per household, these columns may take surprisingly large values for block groups with few households and many empty houses, such as vacation resorts.

It can be downloaded/loaded using the sklearn.datasets.fetch_california_housing function.

- [California Housing Dataset in Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- 20640 samples
- 8 Input Features: 
    - MedInc median income in block group
    - HouseAge median house age in block group
    - AveRooms average number of rooms per household
    - AveBedrms average number of bedrooms per household
    - Population block group population
    - AveOccup average number of household members
    - Latitude block group latitude
    - Longitude block group longitude
- Target: Median house value for California districts, expressed in hundreds of thousands of dollars ($100,000)
  
## Supervised feature selection

### 1. Evaluating mutual info regression method for feature selection

[Mutual information (MI)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html) between two random variables is a non-negative value, which measures the dependency between the variables. It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.

![alt text](https://github.com/sonti-roy/featureSelection_california_housing/blob/main/plots/mutual_info_regression_comparasion.png)

Fig - 1 - The plot show the dependency of target on each feature. 

Top 60% of the features were selected and evaluated for it accuracy with all features dataset using Linear regression.

| Dataset                    | R2       | MSE      |
|----------------------------|----------|----------|
| Original                   | 0.5827018886341137| 0.5468311917368283 |
| Subset                     | 0.5732648513884984 | 0.5591975700714763 |



### 2. Selecting features using f_regression

[f_regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html) uses univariate linear regression tests returning F-statistic and p-values.

![alt text](https://github.com/sonti-roy/featureSelection_california_housing/blob/main/plots/f_regression_comparasion.png)

Fig-2: Plot for F ststistics for all feature against the target.

As the range was highly variable for different features. Top 4 features were selected out of total 8 features and subset data was generated. The score was compared for original and subset dataset.

| Dataset                    | R2       | MSE      |
|----------------------------|----------|----------|
| Original                   | 0.5827018886341137| 0.5468311917368283 |
| subset                     | 0.5732648513884984 | 0.5591975700714763 |


### 3. Pearson Correlation - feature selection

[Pearson correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) is a correlation coefficient that measures linear correlation between two sets of data. It is the ratio between the covariance of two variables and the product of their standard deviations.

![alt text](https://github.com/sonti-roy/featureSelection_california_housing/blob/main/plots/correlation_plot.png)

Longitude and latitude, AveRooms and AveBedrms are highly correlated with -0.92 and 0.85 coeffiecient respectively. For removal of any one feature from the combination variance was analysed for the 4 features.

| Features                   | Variance |
|----------------------------|----------|
| Longitude                  | 4.014139367081251| 
| Latitude                   | 4.562292644202798 | 
| AveRooms                   | 6.12153272384879 | 
| AveBedrms                  | 0.2245915001886127 | 

Based on the variance data, longitude and AveBedrms are removed manually and evaluated the model on original and subset dataset using linear regression.

| Dataset                    | R2       | MSE      |
|----------------------------|----------|----------|
| Original                   | 0.575787706032451| 0.5558915986952441 |
| subset                     | 0.5059804263462322 | 0.6473676847426387 |

### 4. Recurive Feature Elimination (RFE)

[Recursive feature elimination (RFE)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through any specific attribute or callable. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.

Estimator/model used is Lasso as it inheriently do feature selection and subset the dataset and evaluated uisng linear regression model.

| Dataset                    | R2       | MSE      |
|----------------------------|----------|----------|
| Original                   | 0.575787706032451| 0.5558915986952441 |
| subset                     | 0.5059804263462322 | 0.6473676847426387 |

### 5. Sequential Feature Selection

[Sequential Feature Selector](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html) adds (forward selection) or removes (backward selection) features to form a feature subset in a greedy fashion. At each stage, this estimator chooses the best feature to add or remove based on the cross-validation score of an estimator. In the case of unsupervised learning, this Sequential Feature Selector looks only at the features (X), not the desired outputs (y).

RidgeCV estimator is being used for this and selected 6 best features.

| Dataset                    | R2       | MSE      |
|----------------------------|----------|----------|
| Original                   | 0.575787706032451| 0.5558915986952441 |
| subset                     | 0.5059804263462322 | 0.6473676847426387 |

## Unsupervised feature selection

### 1. Principal component analysis

[PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) is defined as an orthogonal linear transformation on a real inner product space that transforms the data to a new coordinate system such that the greatest variance by some scalar projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.

7 component were selected for the transformed space and evalauated it using Linear Regression. 7 was acheived by running at different value and accessing the score.

| Dataset                    | R2       | MSE      |
|----------------------------|----------|----------|
| Original                   | 0.575787706032451| 0.5558915986952441 |
| subset                     | 0.5059804263462322 | 0.6473676847426387 |


## Evaluate different model on the subset of x by multiple models

Around 7 regression model was evaluated on the original dataset set and all the subset dataset and compared their metrics to find the best model with high accuracy.

R2 comparasion for all the model on different subset of data generated through feature selection.
![alt text](https://github.com/sonti-roy/featureSelection_california_housing/blob/main/plots/r2_comparasion_plot.png)

MSE comparasion for all the model on different subset of data generated through feature selection.
![alt text](https://github.com/sonti-roy/featureSelection_california_housing/blob/main/plots/mse_comparasion_plot.png)



| Model                     | R2        | MSE      |
|----------------------------|----------|----------|
| SVR                        | -0.020689| 1.017586 |
| LinearRegression           | 0.582674 | 0.416057 |
| KNeighborsRegression       | 0.136115 | 0.861259 |
| SGDRegressor               | 0.001655 | 0.995310 |
| BayesianRidge              | 0.582681 | 0.416051 |
| DecisionTreeRegressor      | 0.585701 | 0.413039 |
| GradientBoostingRegressor  | 0.772826 | 0.226484 |

*Inference - The model need further tuning to match the score in both the scanerio.*

## Key Takeaways

*How to perform a basic ML model fitting and evaluate the performance of the model.*


## Code 

*The code is is avaiable in a python notebook **<u>model.ipynb</u>**. To view the code please click below*

[*Click here*](https://github.com/sonti-roy/california_housing/blob/main/model.ipynb)

## Roadmap

1. *Model Exploration*
2. *Model Optimization*
3. *Hyperparameter Tuning*
4. *Exploring Other Ways to Improve Model*

## Libraries 

**Language:** Python

**Packages:** Sklearn, Matplotlib, Pandas, Seaborn

## Acknowledgements

*Resources used* 

 - [scikit-learn](https://scikit-learn.org/stable/index.html)
 - OpenAI. (2024). ChatGPT (3.5) Large language model. https://chat.openai.com


## Contact

If you have any feedback/are interested in collaborating, please reach out to me at [LinkdIn](https://www.linkedin.com/in/sonti-roy-phd-8589b711a/)


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

