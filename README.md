![alt text](https://github.com/sonti-roy/california_housing/blob/main/plots/logo.png)


# Project Title
**Feature Selection using supervised and unsupervised method and model development on california housing dataset**


## Implementation Details

- Dataset: California Housing Dataset (view below for more details)
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
| subset                     | 0.5732648513884984 | 0.5591975700714763 |



### 2. Selecting features using f_regression

[f_regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html) uses univariate linear regression tests returning F-statistic and p-values.

![alt text](https://github.com/sonti-roy/featureSelection_california_housing/blob/main/plots/mutual_info_regression_comparasion.png)

1. *Multiple models were evaluated for their performance and compared the R2 and MSE for the models to select the best model.*
   
![alt text](https://github.com/sonti-roy/california_housing/blob/main/plots/model_performance.png)

1. *The performance of GradientBoostingRegressor model was found to be the highest with very low MSEerror compared to other models that are evaluated.*

| Model                     | R2        | MSE      |
|----------------------------|----------|----------|
| SVR                        | -0.020689| 1.017586 |
| LinearRegression           | 0.582674 | 0.416057 |
| KNeighborsRegression       | 0.136115 | 0.861259 |
| SGDRegressor               | 0.001655 | 0.995310 |
| BayesianRidge              | 0.582681 | 0.416051 |
| DecisionTreeRegressor      | 0.585701 | 0.413039 |
| GradientBoostingRegressor  | 0.772826 | 0.226484 |

3. *Model prediction comparasion with true values*

![alt text](https://github.com/sonti-roy/california_housing/blob/main/plots/true_vs_prediction.png)

*Inference - shows a good colinearity which is also visible from the score.*

## Cross valadation

*To evaluate the **<u>GradientBoostingRegressor</u>** model further and check for over fitting, cross valadation is performed.*

1. *Cross validation of the model with complete dataset with cv = 5 shows reduced score than thge model*
        
| Score 1      | Score 2      | Score 3      | Score 4      | Score 5      |
|--------------|--------------|--------------|--------------|--------------|
| 0.62413216   | 0.6943188    | 0.71206383   | 0.65481236   | 0.67672756   |

2. *Cross validation of the model with split dataset shows similar accuracy as the fitted model.*

| Score 1      | Score 2      | Score 3      | Score 4      | Score 5      |
|--------------|--------------|--------------|--------------|--------------|
| 0.78189507   | 0.78282526   | 0.78389246   | 0.80503452   | 0.80055348   |

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

