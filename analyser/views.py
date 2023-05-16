from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.generic import TemplateView # Import TemplateView
import openai
import json
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

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

    # # Scroll to the bottom of the page
    # driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # wait for the JavaScript to finish executing
    time.sleep(5)

    # parse the HTML content with BeautifulSoup
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

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

    data = {
        'result': response.choices[0].text,
        'status': 'success'
    }
    response = JsonResponse(data, status=200, safe=False)
    return response
