# Detailed Report: Linear and Logistic Regression

## 1. Introduction

In this report, we will discuss two important concepts in machine learning: **Linear Regression** and **Logistic Regression**. These algorithms are used for prediction and classification tasks, respectively. Linear Regression is used when the output variable is continuous (like predicting house prices), while Logistic Regression is used for classification (like predicting whether someone will survive or not).

---

## 2. Linear Regression

### 2.1. What is Linear Regression?

Linear Regression is one of the most basic algorithms in machine learning. It is used to model the relationship between a dependent variable (the target) and one or more independent variables (the features). The goal is to find a linear relationship between the input variables and the output variable.

For example, if we have data about house prices, we might want to predict the price of a house based on features like its size, number of rooms, and location. Linear Regression helps us build a model to predict the price from these features.

The formula for **Simple Linear Regression** (with just one feature) is:

$$
y = \beta_0 + \beta_1 x
$$

Where:
- \( y \) is the target (house price),
- \( x \) is the feature (house size),
- $$\( \beta_0 \)$$ is the intercept (where the line crosses the y-axis),
- $$\( \beta_1 \)$$ is the slope of the line (how much the house price changes as the size increases).

In **Multiple Linear Regression**, the formula extends to multiple features:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n
$$

### 2.2. Assumptions of Linear Regression

For Linear Regression to work well, a few assumptions need to be true:
1. **Linearity**: The relationship between the input variables and the output variable should be linear.
2. **Homoscedasticity**: The spread of the residuals (errors) should be the same across all levels of the input variables.
3. **No Multicollinearity**: The independent variables should not be too highly correlated with each other.

### 2.3. How Does Linear Regression Work?

To find the best line (or hyperplane), we need to adjust the model’s parameters \( \beta_0, \beta_1, \dots, \beta_n \) so that the difference between the predicted and actual values is as small as possible. This is done by minimizing the **Mean Squared Error (MSE)**, which is the average of the squared differences between the predicted and actual values.

The algorithm used to find these parameters is **Gradient Descent**. Gradient Descent starts with random values for the parameters and then gradually updates them by taking steps in the direction that reduces the error. The size of the steps is controlled by a value called the **learning rate**.

### 2.4. Regularization

In practice, regularization techniques are often used to improve the performance of Linear Regression models. Regularization helps to prevent **overfitting**, which happens when the model becomes too complex and starts to fit noise in the data.

There are two main types of regularization:
- **L1 Regularization (Lasso)**: Adds a penalty for the absolute values of the coefficients.
- **L2 Regularization (Ridge)**: Adds a penalty for the squared values of the coefficients.

Regularization helps the model generalize better by keeping the coefficients smaller.

For a deeper understanding, you can refer to [this article on Regularization](http://www.holehouse.org/mlclass/07_Regularization.html).

### 2.5. Resources for Further Study on Linear Regression

- [Towards Data Science - Linear Regression Detailed View](https://towardsdatascience.com/linear-regression-detailed-view-ea73175f6e86)
- [ML Cheatsheet - Linear Regression](https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html)
- [Machine Learning Mastery - Simple Linear Regression Tutorial](https://machinelearningmastery.com/simple-linear-regression-tutorial-for-machine-learning/)
- [Holehouse.org - Linear Regression with Multiple Variables](http://www.holehouse.org/mlclass/04_Linear_Regression_with_multiple_variables.html)

---

## 3. Logistic Regression

### 3.1. What is Logistic Regression?

Logistic Regression is used for **classification tasks**, where the goal is to predict which category (or class) something belongs to. Unlike Linear Regression, which is used for predicting continuous values, Logistic Regression is used when the target variable is categorical.

For example, we might want to predict whether a passenger survived or not on the Titanic. Logistic Regression would help us classify a passenger as either “survived” (1) or “not survived” (0).

The logistic function (also called the **sigmoid function**) maps the output of a linear equation to a value between 0 and 1, which can be interpreted as a probability.

The formula for Logistic Regression is:

$$
P(y = 1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n)}}
$$

Where:
- \( P(y = 1 | X) \) is the probability that the target variable \( y \) is 1 (e.g., the passenger survived),
- The rest of the terms are similar to the ones used in Linear Regression.

### 3.2. Cost Function for Logistic Regression

For Logistic Regression, we use a **log loss** (or **cross-entropy loss**) function, which measures how well the model is predicting the classes. The goal is to minimize this cost function using Gradient Descent.

The cost function for Logistic Regression is:

$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_{\theta}(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_{\theta}(x^{(i)})) \right]
$$

Where:
- ![Sigmoid Function](https://latex.codecogs.com/png.latex?h_{\theta}(x^{(i)})) is the predicted probability from the sigmoid function.



### 3.3. Regularization in Logistic Regression

Just like in Linear Regression, regularization can be applied to Logistic Regression to avoid overfitting. **L1** and **L2** regularization are commonly used to penalize large values of the coefficients.

### 3.4. Resources for Further Study on Logistic Regression

- [Towards Data Science - Logistic Regression Detailed Overview](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)
- [Holehouse.org - Logistic Regression](http://www.holehouse.org/mlclass/06_Logistic_Regression.html)
- [ML Cheatsheet - Logistic Regression](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html)
- [Machine Learning Mastery - Logistic Regression for Machine Learning](https://machinelearningmastery.com/logistic-regression-for-machine-learning/)

---

## 4. Conclusion

In this report, we learned about two important algorithms: **Linear Regression** and **Logistic Regression**. Linear Regression is used for predicting continuous values, while Logistic Regression is used for classification tasks. 

We explored how both algorithms work, the assumptions they rely on, and the importance of techniques like **Gradient Descent** and **Regularization**. These algorithms are powerful tools in machine learning, and understanding them is key to solving many real-world problems.

---

## 5. References

- [Towards Data Science - Linear Regression Detailed View](https://towardsdatascience.com/linear-regression-detailed-view-ea73175f6e86)
- [ML Cheatsheet - Linear Regression](https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html)
- [Machine Learning Mastery - Simple Linear Regression Tutorial](https://machinelearningmastery.com/simple-linear-regression-tutorial-for-machine-learning/)
- [Holehouse.org - Linear Regression with Multiple Variables](http://www.holehouse.org/mlclass/04_Linear_Regression_with_multiple_variables.html)
- [Ruder's Optimization Techniques - Gradient Descent](https://ruder.io/optimizing-gradient-descent/)
- [Towards Data Science - Logistic Regression Detailed Overview](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)
- [Holehouse.org - Logistic Regression](http://www.holehouse.org/mlclass/06_Logistic_Regression.html)
