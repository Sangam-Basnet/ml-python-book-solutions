# **Linear Regression**

## **1. Key components of a linear regression equation**  
standard form(simple linear regression):  
      y = b0 + b1x  
Think of this as **"baseline adjustment**  

**Components (memorize this mental model):**  
* y(target/dependent variable)
    -> what you are trying to predict(salary, price, marks, etc.)

* x(input/ independent variable)
  -> what you use to predict(experience, area, hours studied)

* b0 --Intercept
  -> Starting point / default value
    value of y when x = 0

* b1 -Coefficient/ slope
  -> Rate of change
  How much y changes when x increases by 1 unit

**Intercept = where the lie starts**  
**Coefficient = how fast it moves**
  
## 2. Least Squares Method in Linear Regression

The **Least Squares Method** is a mathematical approach used to determine the best-fitting regression line by minimizing the total error between predicted and actual values.

### Objective

Given a dataset with observed values \( y_i \) and predicted values \( \hat{y}_i \), the goal is to minimize the **sum of squared residuals**:

```math
\text{Minimize } \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
