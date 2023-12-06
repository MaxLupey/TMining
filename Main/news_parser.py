import requests
from bs4 import BeautifulSoup
import csv
from tqdm import tqdm

# Відкриваємо файл для запису в CSV
with open('result.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['text', 'target']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Записуємо заголовки
    writer.writeheader()

    for i in tqdm(range(1, 795), desc='Pages', unit='page'):
        # Посилання на сторінку для парсингу
        url = f'https://www.politifact.com/factchecks/?page={i}'

        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Шукаємо всі елементи article
        articles = soup.find_all('article', class_='m-statement')
        for article in tqdm(articles, desc='Processing articles', unit='article'):
            # Отримуємо дані зі статті
            article_text = article.find('div', class_='m-statement__quote').find('a').text
            image_alt = article.find('div', class_='m-statement__content').find('img', class_='c-image__thumb')['alt']

            # Записуємо дані в CSV
            writer.writerow({'text': article_text, 'target': image_alt})

print('Дані були успішно записані в файл result.csv.')