# Stock price prediction using Historical data and Financial news.
TECHNOCOLABS MACHINE LEARNING INTERNSHIP
TITLE:
STOCK PREDICTION USING ML MODELS.
AIM:
Stock price prediction using Historical data and financial news with machine learning models.
INTRODUCTION:
Stock market is trading platform where different investors sale and purchase shares according to stock availability. Stock market ups and downs effects the profit of stakeholders. If market prices going up with available stock, then stakeholders get profit with their purchased stocks. In other case, if market going down with available stock prices, then stakeholders have to face losses. Buyers buy stocks with low prices and sell stocks at high prices and try to get huge profit. Similarly, sellers sell their products at high prices for profit purpose. Stock market (SM) work as trusty platform among sellers and buyers. Advances in Artificial Intelligence (AI) supporting a lot in each field of life with its intelligent features.  Machine learning (ML) is a field of artificial intelligence (AI) that can be considered as we train machines with data and analysis future with test data. SM Prediction provide future trend of stock prices on the basis of previous history. If stakeholders get future predictions, then investment can lead  towards profit. So, for predicting the stock price here, we have used the historical stock prices and historical financial news from Yahoo finance and Nasdaq respectively. Then using this data, we have predicted the stock using different models. The models we have done are:1) Linear Regression,( historical stock prices)
       2)Logistic Regression,( historical stock prices)
       3) Naïve Bayes, (historical financial news)   and
       4)Neural Network(historical stock prices & historical financial news)
In each model we have pre-processing the data with respect to the need of the model and have used the model to predict the stock.
OVERVIEW:
1)	Data Cleaning and pre-processing.
2)	Training the data with various machine learning models.
3)	 Deploying the best model in streamlit using share streamlit.
PREREQUISITES:
You need to install Anaconda python along with the needed libraries like pandas, nltk , sklearn , seaborn , tensorflow , streamlit etc.

TOOLS USED:
Pycharm
Jupyter NoteBook
Google Colab
Spyder
ShareStreamlit
Github
GitBash

DATASETS:
Through web scraping we got the historical stock prices data  and historical financial news data from Yahoo finance and Nasdaq respectively.
Dataset Format: .csv
HistoricalData_APPLE.csv:( historical stock prices data  )
 
News_headlines.csv:( historical financial news data)
 

Stock_Data_with_vader.csv: (historical stock prices & historical financial news)

 

Data visualization for close price and date: 
 

DATA MODELING:
We have used four machine learning models here to predict the stock:
                  1) Linear Regression,( historical stock prices)
                  2)Logistic Regression,( historical stock prices)
                  3) Naïve Bayes, (historical financial news)   and
                  4)Neural Network(historical stock prices & historical financial news)
LINEAR REGRESSION:
Linear regression is used for predictions with data that has numeric target variable. During prediction we use some variables as dependent variables and few considered as independent variables. In situation when there is one dependent and one independent variable, we prefer to use linear regression methodologies. Regression can be single variable or multi variable, it depends upon situation named as single variable or multi variable regression.
 
Here in stock market prediction process, we have one date variable and one closing price variable. Closing price variable is our independent variable which also be considered as target variable. In this processing we will generate a prediction equation using liner regression method. We will generate prediction as y=c+bx, then we can say ‘Y’ is our predicted stock price and x is actual price.

The final data after data pre-processing is:
 

The accuracy score of linear regression model is:
rmse= 0.14598308740936597

r2= 0.9998357614326422
 

 
LOGISTIC REGRESSION:
Logistic Regression is much similar to the Linear Regression except that how they are used. Linear Regression is used for solving Regression problems, whereas Logistic regression is used for solving the classification problems.
 
 In Logistic regression, instead of fitting a regression line, we fit an "S" shaped logistic function, which predicts two maximum values (0 or 1).The curve from the logistic function indicates the likelihood. The S-form curve is called the Sigmoid function or the logistic function.
 
The final data after data pre-processing is:
 
The accuracy score of logistic regression model is:
 
HYPERPARAMETER TUNNING:
As the accuracy is low we use hyperparameter tunning:
We have used GridSearchCV hyperparameter tunning method here,
Tuned Logistic Regression Parameters: {'C': 0.05, 'penalty': 'none'}
Best score is 0.5149105367793241


NAÏVE BAYES:
Naïve Bayes algorithm is a supervised learning algorithm, which is based on Bayes theorem and used for solving classification problems. It is mainly used in text classification that includes a high-dimensional training dataset.
Naïve: It is called Naïve because it assumes that the occurrence of a certain feature is independent of the occurrence of other features. Such as if the fruit is identified on the bases of colour, shape, and taste, then red, spherical, and sweet fruit is recognized as an apple. Hence each feature individually contributes to identify that it is an apple without depending on each other. 
Bayes: It is called Bayes because it depends on the principle of Bayes Theorem. 
 
Where,
P(A|B) is Posterior probability: Probability of hypothesis A on the observed event B.
P(B|A) is Likelihood probability: Probability of the evidence given that the probability of a hypothesis is true.
P(A) is Prior Probability: Probability of hypothesis before observing the evidence.
P(B) is Marginal Probability: Probability of Evidence.
Here we have used The Gaussian model.
The final data after data pre-processing is:
 
The accuracy score of logistic regression model is:
 
HYPERPARAMETER TUNNING:
As the accuracy is low we use hyperparameter tunning:
We have used GridSearchCV hyperparameter tunning method here,
Tuned Logistic Regression Parameters: {'var_smoothing': 1.0}
Best score is 0.5251828631138976

NEURAL NETWORK:
Artificial Neural Network can be best represented as a weighted directed graph, where the artificial neurons form the nodes. The association between the neurons outputs and neuron inputs can be viewed as the directed edges with weights. The Artificial Neural Network receives the input signal from the external source in the form of a pattern and image in the form of a vector. These inputs are then mathematically assigned by the notations x(n) for every n number of inputs. A neuron in a neural network is a mathematical function which collects the data or output from previous layer and classifies the data and give output. This algorithm mimics the operation of the human brain to recognize the patterns in the data.
The final data after data pre-processing is:
 
The accuracy score of logistic regression model is:
 
 
DEPLOYMENT:
Deploying a machine learning model, known as model deployment, simply means to integrate a machine learning model and integrate it into an existing production environment (1) where it can take in an input and return an
output.
The Stock price prediction using Historical data and financial news is trained and tested using Linear Regression Algorithm with 99.98% accuracy. And using streamlit framework, HTML, CSS and python all the necessary files have created along with Procfile file, requirement.txt and setup.sh file.

Using ShareStreamlit, Platform as a service we have successfully deployed our model in the online web server and provided ease to end users.

Deployed Online url:  https://share.streamlit.io/rishitha-8/linear_deploy/main/app.py

 

TEAM MEMBERS:
B Sai Prasanna
B Raja Rishitha
Pranav Krishna Yenni
Appineni Bhanu Prakash 
Sidharth Shankaranarayanan Nai
Dande Aryan Srivatsava
Katta Rakesh
Vithhal Khote

