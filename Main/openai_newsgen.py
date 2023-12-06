import random

import openai
import re
import pandas as pd
openai.api_key = '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'

# Create a response to gpt-4, who creates news
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant journalist."},
        {"role": "user", "content": "You are a journalist bot who writes news, generate 2-3 sentences for each news. Your task is to generate 50 news."}
    ]
)
# get the response choices > message > content
print(response.choices[0].message.content)
news = re.findall(r'\d\.\s(.*)\n', response.choices[0].message.content)
print(news)
for match in news:
    print(match)
    df = pd.read_csv("train.csv")
    new_row = {'text': match, 'target': random.randint(0, 1)}
    df = df._append(new_row, ignore_index=True)
    df.to_csv("train.csv", index=False)