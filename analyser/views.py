from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.generic import TemplateView # Import TemplateView
import openai
import json
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# Read the data from the Excel file
data = pd.read_excel('analyser/datasets/updated_data.xlsx')

# Separate the input features (X) and the target variable (y)
X = data[['Number of tweets', 'Joined Twitter']]
y = data['Number of followers']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the neural network model
activation = 'relu'
solver = 'adam'
early_stopping = True
n_iter_no_change = 50
validation_fraction = 0.1
tol = 0.0001
param_fixed = {'activation':activation, 'solver':solver,'early_stopping':early_stopping, 'n_iter_no_change':n_iter_no_change, 'validation_fraction':validation_fraction,'tol':tol}
hidden_layer_sizes = [[neuron]*hidden_layer for neuron in range(10,60,10) for hidden_layer in range(2,7)]
alpha = [5,1,0.5,0.1,0.05,0.01,0.001]
learning_rate_init = [0.05,0.01,0.005,0.001,0.0005]
beta_1 = [0.85,0.875,0.9,0.95,0.975,0.99,0.995]
beta_2 = [0.99,0.995,0.999,0.9995,0.9999]
param_dist = {'hidden_layer_sizes':hidden_layer_sizes, 'alpha':alpha,'learning_rate_init':learning_rate_init, 'beta_1':beta_1, 'beta_2':beta_2}
NN = MLPRegressor(**param_fixed)
model = RandomizedSearchCV(NN, param_dist, cv=10, verbose=2, n_jobs=-1, n_iter=10, scoring='neg_mean_absolute_error')
model.fit(X_train, y_train)


class HomePageView(TemplateView):
    template_name = "index.html"

def analyze(request):

    # Get the user input
    twitter_account = json.loads(request.body)["account_handler"]

    # set up the webdriver
    driver = webdriver.Chrome()

    # Get the URL of the user's timeline
    url = "https://twitter.com/" + twitter_account

    # load the webpage
    driver.get(url)
    time.sleep(5)

    # Scroll to the bottom of the page
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # wait for the JavaScript to finish executing
    time.sleep(5)

    # parse the HTML content with BeautifulSoup
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    driver.quit()

    # find all the tweets in the page
    divs = soup.find_all("div", {"data-testid": "tweetText"})

    tweets = ""
    # iterate over the tweets and extract the text content
    for div in divs:
        tweets += div.text + "\n\n"
    
    # Set up the OpenAI API credentials
    openai.api_key ="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"

    # Set the model and parameters
    model_engine = "text-davinci-002"
    params = {
        "max_tokens": 200,
        "temperature": 0.9,
    }

    # Generate text using the GPT-3.5 API
    response = openai.Completion.create(
        engine=model_engine,
        prompt="tell me the mental mode of the writer of all these tweets in one sentence: " + tweets,
        max_tokens=params["max_tokens"],
        n=1,
        stop=None,
        temperature=params["temperature"]
    )

    joined_since = 2020 - float(soup.find("span", {'data-testid': 'UserJoinDate'}).text[-4:])

    tweets = soup.find_all("div", {'class': ['css-902oao', 'css-1hf3ou5', 'r-14j79pv', 'r-1k78y06', 'r-n6v787', 'r-16dba41', 'r-1cwl3u0', 'r-bcqeeo', 'r-qvutc0']})
    tweets = float(tweets[8].text.replace(' Tweets',''))
    tweets_ratio = tweets / joined_since
    tweets = tweets + (tweets_ratio * 2)
    
    # Make predictions on the test set
    prediction = int(model.predict([[tweets,joined_since]])[0])

    data = {
        'result': response.choices[0].text,
        'prediction': prediction,
        'status': 'success'
    }
    
    response = JsonResponse(data, status=200, safe=False)
    return response