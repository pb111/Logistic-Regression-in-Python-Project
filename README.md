# Logistic Regression with Python Project


In this project, I implement Logistic Regression algorithm with Python. I build a classifier to predict whether or not it will rain tomorrow in Australia by training a binary classification model using Logistic Regression. I have used the **Rain in Australia** data set downloaded from the Kaggle website for this project.


===============================================================================


## Table of Contents




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


### Decision boundary


The sigmoid function returns a probability value between 0 and 1. This probability value is then mapped to a discrete class which is either “0” or “1”. In order to map this probability value to a discrete class (pass/fail, yes/no, true/false), we select a threshold value. This threshold value is called **Decision boundary**. Above this threshold value, we will map the probability values into class 1 and below which we will map values into class 0.


Mathematically, it can be expressed as follows:-


	p ≥ 0.5 => class = 1
	
	
	p < 0.5 => class = 0 
	

Generally, the decision boundary is set to 0.5. So, if the probability value is 0.8 (> 0.5), we will map this observation to class 1.  Similarly, if the probability value is 0.2 (< 0.5), we will map this observation to class 0.


### Making predictions


Now, we know about `sigmoid function` and `decision boundary` in logistic regression. We can use our knowledge of `sigmoid function` 
and `decision boundary` to write a prediction function. A prediction function in logistic regression returns the probability of the observation being positive, `Yes` or `True`. We call this as `class 1` and it is denoted by `P(class = 1)`. If the probability inches closer to one, then we will be more confident about our model that the observation is in class 1.


In the previous example, suppose the sigmoid function returns the probability value of 0.4. It means that there is only 40% chance of passing the exam. If the decision boundary is 0.5, then we predict this observation as `fail`.


### Cost function


In this case, the prediction function is non-linear due to the sigmoid transformation. We square this prediction function to get the mean square error (MSE). It results in a non-convex function with many local minimums. If the cost function has many local minimums, then the gradient descent may not converge and do not find the global optimal minimum. So, instead of mean square error (MSE), we use a cost-function called **Cross-Entropy**.


### Cross-Entropy


**Cross-Entropy** is a cost-function which measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-Entropy loss is also known as **Log Loss**. It can be divided into two separate cost-functions: one for y = 1 and one for y = 0. 


Mathematically, it can be given with the following formula.


**Formula**


The cross-entropy loss function can be represented with the following graphs for y = 1 and y = 0. These are smooth monotonic functions which always increases or always decreases. They help us to easily calculate the gradient and minimizing cost. 


**Image - Cross entropy cost function from the Andrew Ng’s slides**


Cross-entropy loss increases as the predicted probability diverges from the actual label. This cost-function penalizes confident and wrong predictions more than it rewards confident and right predictions. A perfect model would have a log loss of zero.


The above loss-functions can be compressed into one function as follows.


**Image**


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


**Image**



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



