# Logistic Regression with Python and Scikit-Learn 


In this project, I implement Logistic Regression algorithm with Python. I build a classifier to predict whether or not it will rain tomorrow in Australia by training a binary classification model using Logistic Regression. I have used the **Rain in Australia** data set downloaded from the Kaggle website for this project.


===============================================================================


## Table of Contents


1.	Introduction to Logistic Regression
2.	Linear Regression vs Logistic Regression
3.	Logistic Regression intuition
4.	Assumptions of Logistic Regression
5.	Types of Logistic Regression
6.	The problem statement
7.	Results and conclusion
8.	References


===============================================================================


## 1. Introduction to Logistic Regression


When data scientists may come across a new classification problem, the first algorithm that may come across their mind is **Logistic Regression**. It is a supervised learning classification algorithm which is used to predict observations to a discrete set of classes. Practically, it is used to classify observations into different categories. Hence, its output is discrete in nature. **Logistic Regression** is also called **Logit Regression**. It is one of the most simple, straightforward and versatile classification algorithms which is used to solve classification problems.


People who are new to machine learning may get confused with the Logistic Regression name. They tend to think that it is a regression algorithm. But, this is not true. Logistic Regression is the classification algorithm and it is used for supervised learning classification problems. 


So, I will start the discussion by comparing differences between Linear Regression and Logistic Regression.


===============================================================================


## 2. Linear Regression vs Logistic Regression


In this section, I will elaborate the differences between Linear Regression and Logistic Regression. The differences are listed below:-

1.	Linear regression is used to predict continuous outputs whereas Logistic Regression is used to predict discrete set of outputs which is mapped to different classes. 


2.	So, the examples of Linear Regression are predicting the house prices and stock prices. The examples of Logistic Regression include predicting whether a student will fail or pass and whether a patient will survive or not after a major operation.


3.	Linear Regression is based on Ordinary Least Squares (OLS) estimation whereas Logistic Regression is based on Maximum Likelihood Estimation (MLE) approach.


   The difference between Linear Regression and Logistic Regression can be represented diagrammatically as follows-
   
   
   ![Linear Regression Vs Logistic Regression](https://github.com/pb111/Logistic-Regression-in-Python-Project/blob/master/Images/Linear%20Regression%20Vs%20Logistic%20Regression.jpeg)
   
   
===============================================================================


## 3. Logistic Regression intuition


In statistics, the **Logistic Regression model** is a widely used statistical model which is primarily used for classification purposes. It means that given a set of observations, Logistic Regression algorithm helps us to classify these observations into two or more discrete classes. So, the target variable is discrete in nature.


The Logistic Regression algorithm works as follows:-


### Implement linear equation


Logistic Regression algorithm works by implementing a linear equation with independent or explanatory variables to predict a response value. For example, we consider the example of number of hours studied and probability of passing the exam. Here, number of hours studied is the explanatory variable and it is denoted by x1. Probability of passing the exam is the response or target variable and it is denoted by z. 


If we have one explanatory variable (x1) and one response variable (z), then the linear equation would be given mathematically with the following equation- 


 	z = β0 + β1x1


Here, the coefficients β0 and β1 are the parameters of the model.


If there are multiple explanatory variables, then the above equation can be extended to 


	z = β0 + β1x1+ β2x2+……..+ βnxn


Here, the coefficients β0, β1, β2 and βn are the parameters of the model.


So, the predicted response value is given by the above equations and is denoted by z.

 
### Sigmoid Function


This predicted response value, denoted by z is then converted into a probability value that lie between 0 and 1. We use the **sigmoid function** in order to map predicted values to probability values. This sigmoid function then maps any real value into a probability value between 0 and 1. 


In machine learning, sigmoid function is used to map predictions to probabilities. The sigmoid function has an `S` shaped curve. It is also called **sigmoid curve**.  


A `Sigmoid function` is a special case of the `Logistic function`. It is given by the following mathematical formula. 


Graphically, we can represent sigmoid function with the following graph.



![Sigmoid function](https://github.com/pb111/Logistic-Regression-in-Python-Project/blob/master/Images/Sigmoid%20function.png)



### Decision boundary


The sigmoid function returns a probability value between 0 and 1. This probability value is then mapped to a discrete class which is either “0” or “1”. In order to map this probability value to a discrete class (pass/fail, yes/no, true/false), we select a threshold value. This threshold value is called **Decision boundary**. Above this threshold value, we will map the probability values into class 1 and below which we will map values into class 0.


Mathematically, it can be expressed as follows:-


	p ≥ 0.5 => class = 1
	
	
	p < 0.5 => class = 0 
	

Generally, the decision boundary is set to 0.5. So, if the probability value is 0.8 (> 0.5), we will map this observation to class 1.  Similarly, if the probability value is 0.2 (< 0.5), we will map this observation to class 0.


![Decision boundary](https://github.com/pb111/Logistic-Regression-in-Python-Project/blob/master/Images/Logistic%20Regression_Decision%20Boundary.png)


### Making predictions


Now, we know about `sigmoid function` and `decision boundary` in logistic regression. We can use our knowledge of `sigmoid function` 
and `decision boundary` to write a prediction function. A prediction function in logistic regression returns the probability of the observation being positive, `Yes` or `True`. We call this as `class 1` and it is denoted by `P(class = 1)`. If the probability inches closer to one, then we will be more confident about our model that the observation is in class 1.


In the previous example, suppose the sigmoid function returns the probability value of 0.4. It means that there is only 40% chance of passing the exam. If the decision boundary is 0.5, then we predict this observation as `fail`.


### Cost function


In this case, the prediction function is non-linear due to the sigmoid transformation. We square this prediction function to get the mean square error (MSE). It results in a non-convex function with many local minimums. If the cost function has many local minimums, then the gradient descent may not converge and do not find the global optimal minimum. So, instead of mean square error (MSE), we use a cost-function called **Cross-Entropy**.


### Cross-Entropy


**Cross-Entropy** is a cost-function which measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-Entropy loss is also known as **Log Loss**. It can be divided into two separate cost-functions: one for y = 1 and one for y = 0. 


Mathematically, it can be given with the following formula.


![Cross Entropy Formula](https://github.com/pb111/Logistic-Regression-in-Python-Project/blob/master/Images/Cross%20Entropy.png)


The cross-entropy loss function can be represented with the following graphs for y = 1 and y = 0. These are smooth monotonic functions which always increases or always decreases. They help us to easily calculate the gradient and minimizing cost. 


![Cross entropy loss function](https://github.com/pb111/Logistic-Regression-in-Python-Project/blob/master/Images/Cross%20entropy%20loss%20function.png)


Cross-entropy loss increases as the predicted probability diverges from the actual label. This cost-function penalizes confident and wrong predictions more than it rewards confident and right predictions. A perfect model would have a log loss of zero.


The above loss-functions can be compressed into one function as follows.


![Compressed logistic cost-function](https://github.com/pb111/Logistic-Regression-in-Python-Project/blob/master/Images/Compressed_logistic_cost_function.png)


In binary classification models, where the number of classes is equal to 2, cross-entropy can be calculated as follows.


		-(y log (p) + (1 – y) log (1 – p))


If there is a multiclass classification, we calculate a separate loss for each class label per observation and sum the result as follows.


		-Ʃ yo,c log (po,c) 



Here, 

-	Summation is over the number of classes.

-	log – natural logarithm

-	y – binary indicator (0 or 1) if class label c is correctly classified for observation o.

-	predicted probability observation o is of class C.


 
**Vectorized cost function** can be given as follows.


![Vectorized cost function](https://github.com/pb111/Logistic-Regression-in-Python-Project/blob/master/Images/Vectorized%20cost%20function.png)



### Gradient descent


To minimize the cost-function, we use **gradient descent** technique. Python machine-learning library Scikit-learn hide this implementation.


The derivative of the sigmoid function is given by the following formula.


	 	sʹ(z) = s(z) (1 - s(z))
		

The above equation leads us to the cost-function given by the following formula.


		Cʹ = x (s(z) - y)
		

Here, we have


-	C′ is the derivative of cost with respect to weights


-	y is the actual class label (0 or 1)


-	s(z) is the model prediction


-	x is the feature vector


### Mapping probabilities to classes


The final step is to assign class labels (0 or 1) to the predicted probabilities.


===============================================================================


## 4. Assumptions of Logistic Regression


Logistic Regression does not require the key assumptions of linear regression and generalized linear models. In particular, it does not require the following key assumptions of linear regression:-


1.	Logistic Regression does not follow the assumption of linearity. It does not require a linear relationship between the independent and dependent variables. 


2.	The residuals or error terms do not need to follow the normal distribution.


3.	Logistic Regression does not require the assumption of homoscedasticity. Homoscedasticity means all the variables in the model have same variance. So, in Logistic Regression model, the variables may have different variance.


4.	The dependent variable in Logistic Regression is not measured on an interval or ratio scale.


The Logistic Regression model requires several key assumptions. These are as follows:-


1.	Logistic Regression model requires the dependent variable to be binary, multinomial or ordinal in nature. 


2.	It requires the observations to be independent of each other. So, the observations should not come from repeated measurements.


3.	Logistic Regression algorithm requires little or no multicollinearity among the independent variables. It means that the independent variables should not be too highly correlated with each other.


4.	Logistic Regression model assumes linearity of independent variables and log odds.


5.	The success of Logistic Regression model depends on the sample sizes. Typically, it requires a large sample size to achieve the high accuracy.


===============================================================================


## 5. Types of Logistic Regression


Logistic Regression model can be classified into three groups based on the target variable categories. These three groups are described below:-


1.	**Binary Logistic Regression**


In **Binary Logistic Regression**, the target variable has two possible categories. The common examples of categories are `yes or no`, `good or bad`, `true or false`, `spam or no spam` and `pass or fail`.


2.	**Multinomial Logistic Regression**


In **Multinomial Logistic Regression**, the target variable has three or more categories which are not in any particular order. So, there are three or more nominal categories. The examples include the type of categories of fruits - `apple`, `mango`, `orange` and `banana`.


3.	**Ordinal Logistic Regression**


In **Ordinal Logistic Regression**, the target variable has three or more ordinal categories. So, there is intrinsic order involved with the categories. For example, the student performance can be categorized as `poor`, `average`, `good` and `excellent`.


===============================================================================


## 6. The problem statement


In this project, I try to answer the question that whether or not it will rain tomorrow in Australia. I implement Logistic Regression with Python and Scikit-Learn. 


To answer the question, I build a classifier to predict whether or not it will rain tomorrow in Australia by training a binary classification model using Logistic Regression. I have used the **Rain in Australia** dataset downloaded from the Kaggle website for this project.


The Python implementation is presented in the Jupyter notebook


===============================================================================


## 7. Results and conclusion


1.	The logistic regression model accuracy score is 0.8501. So, the model does a very good job in predicting whether or not it will rain tomorrow in Australia.


2.	Small number of observations predict that there will be rain tomorrow. Majority of observations predict that there will be no rain tomorrow.


3.	The model shows no signs of overfitting.


4.	Increasing the value of C results in higher test set accuracy and also a slightly increased training set accuracy. So, we can conclude that a more complex model should perform better.


5.	Increasing the threshold level results in increased accuracy.


6.	ROC AUC of our model approaches towards 1. So, we can conclude that our classifier does a good job in predicting whether it will rain tomorrow or not.


7.	Our original model accuracy score is 0.8501 whereas accuracy score after RFECV is 0.8500. So, we can obtain approximately similar accuracy but with reduced set of features.


8.	In the original model, we have FP = 1175 whereas FP1 = 1174. So, we get approximately same number of false positives. Also, FN = 3087 whereas FN1 = 3091. So, we get slighly higher false negatives.


9.	Our, original model score is found to be 0.8476. The average cross-validation score is 0.8474. So, we can conclude that cross-validation does not result in performance improvement.


10.	Our original model test accuracy is 0.8501 while GridSearch CV accuracy is 0.8507.
We can see that GridSearch CV improve the performance for this particular model.


===============================================================================


## 8. References

The work done in this project is inspired from following books and websites:-


1.	Hands on Machine Learning with Scikit-Learn and Tensorflow by Aurélién Géron


2.	Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido


3.	Udemy course – Machine Learning – A Z by Kirill Eremenko and Hadelin de Ponteves


4.	Udemy course – Feature Engineering for Machine Learning by Soledad Galli


5.	Udemy course – Feature Selection for Machine Learning by Soledad Galli


6.	https://en.wikipedia.org/wiki/Logistic_regression


7.	https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html


8.	https://en.wikipedia.org/wiki/Sigmoid_function


9.	https://www.statisticssolutions.com/assumptions-of-logistic-regression/


10.	https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python


11.	https://www.kaggle.com/neisha/heart-disease-prediction-using-logistic-regression


12.	https://www.ritchieng.com/machine-learning-evaluate-classification-model/














