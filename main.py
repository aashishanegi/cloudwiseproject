import csv
from flask import Flask, render_template
import requests
from transformers import pipeline, AutoTokenizer
import os
import pandas as pd
import torch
from torch import Tensor

import yfinance as yf  # Import yfinance for Indian stock prices

app = Flask(__name__)

# Load summarizer pipeline and tokenizer
model_name = 'sshleifer/distilbart-cnn-12-6'
summarizer = pipeline("summarization", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to ensure the input text is valid and truncated to 1000 tokens if necessary
def process_text_for_summarization(text):
    if not isinstance(text, str):
        # Handle case where text is not a string (e.g., it's a float or None)
        print("Error: Text is not a string")
        return None

    # Tokenize the text
    inputs = tokenizer.encode(text, return_tensors='pt', truncation=False)
    input_length = inputs.size(1)  # Sequence length

    if input_length > 1024:
        # Truncate the input_ids to 1000 tokens
        inputs = inputs[:, :1000]
        # Convert back to text
        text = tokenizer.decode(inputs[0], skip_special_tokens=True)
        print("Text was truncated to 1000 tokens.")

    return text

# Function to summarize the text safely, trimming if necessary
def safe_summarization(text):
    try:
        # Process the text to ensure it's valid and within the 1024 token limit
        processed_text = process_text_for_summarization(text)
        
        if not processed_text:
            return text  # If text is invalid (None), return the original content

        # Tokenize the processed text to get input length in tokens
        inputs = tokenizer.encode(processed_text, return_tensors='pt', truncation=False)
        input_length = inputs.size(1)

        # If input length is less than 200 tokens, skip summarization
        if input_length < 200:
            return processed_text  # Return full text if not summarized

        # Set max_length and min_length for summarization
        max_length = min(150, int(input_length * 0.3))
        min_length = min(50, int(input_length * 0.1))

        # Perform summarization
        summary = summarizer(processed_text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    
    except Exception as e:
        print(f"Error summarizing: {e}")
        # Return original article content in case of any summarization error
        return text

# Function to summarize articles from a CSV file
def summarize_news(csv_file, source_name):
    summarized_articles = []
    try:
        news_data = pd.read_csv(csv_file)
        for _, row in news_data.iterrows():
            headline = row['title']
            article = row['article']
            # Safely summarize the article with error handling
            summary = safe_summarization(article)
            summarized_articles.append({'headline': headline, 'summary': summary, 'link': row['link'], 'source': source_name})
    except Exception as e:
        print(f"Error summarizing news from {csv_file}: {e}")
    return summarized_articles

# Function to fetch free weather data using Open-Meteo API
def fetch_weather(location="New Delhi"):
    try:
        # Coordinates for New Delhi
        latitude = 28.6139
        longitude = 77.2090
        # Open-Meteo API URL for fetching weather data
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current_weather": True
            }
        )
        data = response.json()
        weather_code_mapping = {
            0: 'Clear sky',
            1: 'Mainly clear',
            2: 'Partly cloudy',
            3: 'Overcast',
            45: 'Fog',
            48: 'Depositing rime fog',
            51: 'Light drizzle',
            53: 'Moderate drizzle',
            55: 'Dense drizzle',
            56: 'Light freezing drizzle',
            57: 'Dense freezing drizzle',
            61: 'Slight rain',
            63: 'Moderate rain',
            65: 'Heavy rain',
            66: 'Light freezing rain',
            67: 'Heavy freezing rain',
            71: 'Slight snow fall',
            73: 'Moderate snow fall',
            75: 'Heavy snow fall',
            77: 'Snow grains',
            80: 'Slight rain showers',
            81: 'Moderate rain showers',
            82: 'Violent rain showers',
            85: 'Slight snow showers',
            86: 'Heavy snow showers',
            95: 'Thunderstorm',
            96: 'Thunderstorm with slight hail',
            99: 'Thunderstorm with heavy hail',
        }
        condition_code = data['current_weather']['weathercode']
        condition = weather_code_mapping.get(condition_code, 'Unknown')
        weather_info = {
            'location': location,
            'temperature': data['current_weather']['temperature'],
            'condition': condition,
            'windspeed': data['current_weather']['windspeed']
        }
        return weather_info
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return {'location': location, 'temperature': 'N/A', 'condition': 'N/A', 'windspeed': 'N/A'}

# Function to fetch Indian gold prices using GoldAPI.io
def fetch_gold_prices():
    try:
        api_key = 'goldapi-653c19m29j3ssy-io'  # Replace with your actual GoldAPI.io API key
        headers = {
            'x-access-token': api_key,
            'Content-Type': 'application/json'
        }
        # Fetch gold price in INR
        response_gold = requests.get('https://www.goldapi.io/api/XAU/INR', headers=headers)
        data_gold = response_gold.json()
        
        if response_gold.status_code == 200:
            # Extract price per gram of 24k gold and calculate price per 10 grams
            price_per_gram_24k = data_gold.get('price_gram_24k', 'N/A')
            if price_per_gram_24k != 'N/A':
                gold_price_per_10g = float(price_per_gram_24k) * 10
                gold_price_per_10g = f"{gold_price_per_10g:.2f}"
            else:
                gold_price_per_10g = 'N/A'
            
            # Extract price per gram of 999 silver and calculate price per 10 grams
            response_silver = requests.get('https://www.goldapi.io/api/XAG/INR', headers=headers)
            data_silver = response_silver.json()
            if response_silver.status_code == 200:
                price_per_gram_silver = data_silver.get('price_gram_24k', 'N/A')
                if price_per_gram_silver != 'N/A':
                    silver_price_per_10g = float(price_per_gram_silver) * 10
                    silver_price_per_10g = f"{silver_price_per_10g:.2f}"
                else:
                    silver_price_per_10g = 'N/A'
            else:
                silver_price_per_10g = 'N/A'
            
            return {'gold': gold_price_per_10g, 'silver': silver_price_per_10g}
        else:
            error_message = data_gold.get('error', {}).get('message', 'Unknown error')
            print(f"Error fetching gold prices: {error_message}")
            return {'gold': 'N/A', 'silver': 'N/A'}
    except Exception as e:
        print(f"Exception occurred fetching gold prices: {e}")
        return {'gold': 'N/A', 'silver': 'N/A'}

# Function to fetch multiple Indian stock prices using yfinance
def fetch_stock_prices(symbols=None):
    if symbols is None:
        symbols = ["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS"]  # Removed indices

    stock_data_list = []

    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period='5d', interval='1d')  # Valid period and interval
            if not data.empty and len(data) >= 2:
                latest_close = data['Close'].iloc[-1]
                previous_close = data['Close'].iloc[-2]
                price_change = latest_close - previous_close
                price_change_percent = (price_change / previous_close) * 100
                # Determine the arrow symbol
                if price_change > 0:
                    arrow = "🡅"
                elif price_change < 0:
                    arrow = "🡇"
                else:
                    arrow = "-"
                stock_data_list.append({
                    'symbol': symbol,
                    'name': stock.info.get('shortName', symbol),
                    'price': f"{latest_close:.2f}",
                    'change': f"{price_change:.2f}",
                    'percent_change': f"{price_change_percent:.2f}%",
                    'arrow': arrow
                })
            else:
                print(f"No data found for the stock symbol: {symbol}")
                stock_data_list.append({
                    'symbol': symbol,
                    'name': symbol,
                    'price': 'N/A',
                    'change': 'N/A',
                    'percent_change': 'N/A',
                    'arrow': '-'
                })
        except Exception as e:
            print(f"Error fetching stock prices for {symbol}: {e}")
            stock_data_list.append({
                'symbol': symbol,
                'name': symbol,
                'price': 'N/A',
                'change': 'N/A',
                'percent_change': 'N/A',
                'arrow': '-'
            })
    return stock_data_list


# Flask route to display data
@app.route('/')
def index():
    # Summarize news from different channels
    news_sources = {
        'hindu': {'file': 'hindu_articles.csv', 'source': 'The Hindu'},
        'et': {'file': 'ET_articles.csv', 'source': 'Economic Times'},
        'bbc': {'file': 'bbc_articles.csv', 'source': 'BBC News'},
        'toi': {'file': 'IE_articles.csv', 'source': 'Indian Express'}
    }

    summarized_news_by_source = {}
    for key, value in news_sources.items():
        summarized_news_by_source[key] = summarize_news(value['file'], value['source'])

    # Fetch weather, gold prices, and stock prices
    weather_data = fetch_weather(location="New Delhi")
    gold_prices = fetch_gold_prices()
    stock_symbols = ["^NSEI", "^BSESN", "RELIANCE.NS", "HDFCBANK.NS", "TCS.NS"]
    stock_data_list = fetch_stock_prices(symbols=stock_symbols)

    return render_template(
        'index.html', 
        summarized_news_by_source=summarized_news_by_source,
        weather=weather_data, 
        gold=gold_prices, 
        stocks=stock_data_list
    )

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5001)
