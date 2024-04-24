https://colab.research.google.com/drive/1YxJUde8XWFj5I8yTlg4a_gOE7J19rnwV?usp=sharing

# ML-Project---Yes-Bank-Stock-Closing-Price-Prediction

# (Supervised - Regression)

# Contribution :- Individual

# Name :- Mohan chhangani

images (1).jfif

# Problem Statement

Yes Bank is a well-known bank in the Indian financial domain. Since 2018, it has been in the news because of the fraud case involving Rana Kapoor. Owing to this fact, it was interesting to see how that impacted the stock prices of the company and whether Time series models or any other predictive models can do justice to such situations. This dataset has monthly stock prices of the bank since its inception and includes closing, starting, highest, and lowest stock prices of every month. The main objective is to predict the stock’s closing price of the month.

# mount drive to load dataset
from google.colab import drive
drive.mount('/content/drive')

# importing the required libraries

# Import Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings('ignore')

Dataset Loading

import datetime
