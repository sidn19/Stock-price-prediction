# Stock price prediction using Historical data and Financial news.
**TECHNOCOLABS MACHINE LEARNING INTERNSHIP**


**TITLE:**

STOCK PREDICTION USING ML MODELS.


**AIM:**

Stock price prediction using Historical data and financial news with machine learning models.


**INTRODUCTION:**

Stock market is trading platform where different investors sale and purchase shares according to stock availability. Stock market ups and downs effects the profit of stakeholders. If market prices going up with available stock, then stakeholders get profit with their purchased stocks. In other case, if market going down with available stock prices, then stakeholders have to face losses. Buyers buy stocks with low prices and sell stocks at high prices and try to get huge profit. Similarly, sellers sell their products at high prices for profit purpose. Stock market (SM) work as trusty platform among sellers and buyers. Advances in Artificial Intelligence (AI) supporting a lot in each field of life with its intelligent features.  Machine learning (ML) is a field of artificial intelligence (AI) that can be considered as we train machines with data and analysis future with test data. SM Prediction provide future trend of stock prices on the basis of previous history. If stakeholders get future predictions, then investment can lead  towards profit. So, for predicting the stock price here, we have used the historical stock prices and historical financial news from Yahoo finance and Nasdaq respectively. Then using this data, we have predicted the stock using different models. The models we have done are:
        1)**** Linear Regression****,( historical stock prices)
       2)**Logistic Regression**,( historical stock prices)
       3) **Naïve Baye**s, (historical financial news)   and
       4)**Neural Network**(historical stock prices & historical financial news)
In each model we have pre-processing the data with respect to the need of the model and have used the model to predict the stock.


**OVERVIEW:**
1)	Data Cleaning and pre-processing.
2)	Training the data with various machine learning models.
3)	 Deploying the best model in streamlit using share streamlit.


**PREREQUISITES:**

You need to install Anaconda python along with the needed libraries like pandas, nltk , sklearn , seaborn , tensorflow , streamlit etc.

**TOOLS USED:**
Pycharm

Jupyter NoteBook

Google Colab

Spyder

ShareStreamlit

Github

GitBash

**DATASETS:**

Through web scraping we got the historical stock prices data  and historical financial news data from Yahoo finance and Nasdaq respectively.

Dataset Format: **.csv**

**HistoricalData_APPLE.csv**:( historical stock prices data  )

 ![1](https://user-images.githubusercontent.com/79561409/125926016-fc601725-28d2-47d9-881b-483c011ba0ed.png)
 
**News_headlines.csv:**( historical financial news data)

 ![2](https://user-images.githubusercontent.com/79561409/125926026-96070c5d-2415-483b-928d-5cf5736dfe91.png)

**Stock_Data_with_vader.csv:** (historical stock prices & historical financial news)

![3](https://user-images.githubusercontent.com/79561409/125926034-3a4a094e-04a9-4e11-ab4d-ac588b3507d7.png)

Data visualization for close price and date: 
 
 ![4](https://user-images.githubusercontent.com/79561409/125926054-b00d6626-9c48-4001-a0c1-7a7dba2baa82.png)
 
**DATA MODELING:**

We have used four machine learning models here to predict the stock:
                  1) **Linear Regression**,( historical stock prices)
                  2)**Logistic Regression**,( historical stock prices)
                  3) **Naïve Bayes**, (historical financial news)   and
                  4)**Neural Network**(historical stock prices & historical financial news
                  
  
  
**LINEAR REGRESSION:**

Linear regression is used for predictions with data that has numeric target variable. During prediction we use some variables as dependent variables and few considered as independent variables. In situation when there is one dependent and one independent variable, we prefer to use linear regression methodologies. Regression can be single variable or multi variable, it depends upon situation named as single variable or multi variable regression.

 ![5](https://user-images.githubusercontent.com/79561409/125926062-bab7ad3f-98b3-401f-9a71-32bd35ac575a.png)
 
Here in stock market prediction process, we have one date variable and one closing price variable. Closing price variable is our independent variable which also be considered as target variable. In this processing we will generate a prediction equation using liner regression method. We will generate prediction as **y=c+bx**, then we can say ‘Y’ is our predicted stock price and x is actual price.

The final data after data pre-processing is:
 ![6](https://user-images.githubusercontent.com/79561409/125926093-81b8eec8-3506-42f9-93a8-3db506a55945.png)

The accuracy score of linear regression model is:

rmse= 0.14598308740936597
r2= 0.9998357614326422

 ![7](https://user-images.githubusercontent.com/79561409/125926112-89e833ba-22e4-41b4-830a-533ff95533f5.png)
 
![8](https://user-images.githubusercontent.com/79561409/125926124-ac58b767-6cdc-4983-9e82-478c3855fe4f.png)



**LOGISTIC REGRESSION:**

Logistic Regression is much similar to the Linear Regression except that how they are used. Linear Regression is used for solving Regression problems,**** whereas Logistic regression is used for solving the classification problems.***

 ![9](https://user-images.githubusercontent.com/79561409/125926140-45568b34-7fcd-4c4a-bbdb-8dfe49d06fa7.png)
 
 In Logistic regression, instead of fitting a regression line, we fit an "S" shaped logistic function, which predicts two maximum values (0 or 1).The curve from the logistic function indicates the likelihood. The S-form curve is called the Sigmoid function or the logistic function.
 
 ![10](https://user-images.githubusercontent.com/79561409/125926154-15c62512-1de3-45b7-a0a7-3b5e6a610939.png)
 
The final data after data pre-processing is:

![11](https://user-images.githubusercontent.com/79561409/125926160-3640c714-0d94-4062-a88c-cc7bc2ff5a40.png)

The accuracy score of logistic regression model is:

 ![12](https://user-images.githubusercontent.com/79561409/125926167-fc0bc068-dacf-4218-9143-a1e362007080.png)
 
HYPERPARAMETER TUNNING:

As the accuracy is low we use hyperparameter tunning:
We have used GridSearchCV hyperparameter tunning method here,
Tuned Logistic Regression Parameters: {'C': 0.05, 'penalty': 'none'}
Best score is 0.5149105367793241



**NAÏVE BAYES:**

Naïve Bayes algorithm is a supervised learning algorithm, which is based on **Bayes theorem** and used for solving classification problems. It is mainly used in text classification that includes a high-dimensional training dataset.
**Naïve:** It is called Naïve because it assumes that the occurrence of a certain feature is independent of the occurrence of other features. Such as if the fruit is identified on the bases of colour, shape, and taste, then red, spherical, and sweet fruit is recognized as an apple. Hence each feature individually contributes to identify that it is an apple without depending on each other. 
**Bayes:** It is called Bayes because it depends on the principle of Bayes Theorem. 

 ![13](https://user-images.githubusercontent.com/79561409/125926176-8edabbdb-b890-4980-8371-5de286a2bbec.png)
 
Where,
**P(A|B) is Posterior probability:** Probability of hypothesis A on the observed event B.

**P(B|A) is Likelihood probability:** Probability of the evidence given that the probability of a hypothesis is true.

**P(A) is Prior Probability:** Probability of hypothesis before observing the evidence.

**P(B) is Marginal Probability:** Probability of Evidence.

Here we have used The Gaussian model.

The final data after data pre-processing is:

 ![14](https://user-images.githubusercontent.com/79561409/125926188-38c64952-c81c-4a38-a659-906bf4a15995.png)
 
The accuracy score of logistic regression model is:

 ![15](https://user-images.githubusercontent.com/79561409/125926199-85571f2c-07db-48bc-9de1-87419c38bf7d.png)
 
HYPERPARAMETER TUNNING:
As the accuracy is low we use hyperparameter tunning:
We have used GridSearchCV hyperparameter tunning method here,
Tuned Logistic Regression Parameters: {'var_smoothing': 1.0}
Best score is 0.5251828631138976



**NEURAL NETWORK:***

Artificial Neural Network can be best represented as a weighted directed graph, where the artificial neurons form the nodes. The association between the neurons outputs and neuron inputs can be viewed as the directed edges with weights. The Artificial Neural Network receives the input signal from the external source in the form of a pattern and image in the form of a vector. These inputs are then mathematically assigned by the notations x(n) for every n number of inputs. A neuron in a neural network is a mathematical function which collects the data or output from previous layer and classifies the data and give output. This algorithm mimics the operation of the human brain to recognize the patterns in the data.

The final data after data pre-processing is:

 ![16](https://user-images.githubusercontent.com/79561409/125926202-0cf32d6b-9a56-445e-a18e-24ee8ddce4e1.png)
 
The accuracy score of Neural Network model is:

 ![17](https://user-images.githubusercontent.com/79561409/125926209-50bac20e-62f0-4f21-b003-6d85178d3eeb.png)
 
 ![18](https://user-images.githubusercontent.com/79561409/125926215-f4ea0515-a3da-4e24-913e-901e9c22d81e.png)
 
 
 
 
 
**DEPLOYMENT:**

Deploying a machine learning model, known as model deployment, simply means to integrate a machine learning model and integrate it into an existing production environment (1) where it can take in an input and return an output.

The Stock price prediction using Historical data and financial news is trained and tested using Linear Regression Algorithm with 99.98% accuracy. And using streamlit framework, HTML, CSS and python all the necessary files have created along with Procfile file, requirement.txt and setup.sh file.

Using ShareStreamlit, Platform as a service we have successfully deployed our model in the online web server and provided ease to end users.

Deployed Online url:  https://share.streamlit.io/rishitha-8/linear_deploy/main/app.py

 ![19](https://user-images.githubusercontent.com/79561409/125926230-357d71e3-23fa-4a4d-93bf-34065932e86a.png)

**TEAM MEMBERS:**

B Sai Prasanna

B Raja Rishitha

Pranav Krishna Yenni

Appineni Bhanu Prakash

Sidharth Shankaranarayanan Nair


Dande Aryan Srivatsava

Katta Rakesh

Vithhal Khote

