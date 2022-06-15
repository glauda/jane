from bs4 import BeautifulSoup
from datetime import date, datetime, timedelta
import pandas as pd
import random
import requests
from typing import Dict, List
import unicodedata


def create_extracted_dataset(
        dt_start: date,
        dt_end: date,
        nb_iteration: int,
        l_base_url: List[str],
        seed: int) -> pd.DataFrame:
    """ Extract medium articles from random days between a time interval in a certain blogs

    Args:
        dt_start: Beginning of the range time
        dt_end: End of the range time
        nb_iteration: Number of random days to extract
        l_base_url: List of medium blogs to search in
        seed: random seed to make pipeline idempotent

    Returns:
        Dataframe of extracted articles
    """

    print(
        f'--- Node input arguments ---\n'
        f'dt_start: {dt_start}\n'
        f'dt_end: {dt_end}\n'
        f'nb_iteration: {nb_iteration}\n'
        f'l_base_url: {l_base_url}\n'
        f'seed: {seed}\n'
        f'------'
    )

    # Setup variables
    # dt_start = datetime.strptime(start_date, '%Y-%m-%d')
    # dt_end = datetime.strptime(end_date, '%Y-%m-%d')
    df_article = pd.DataFrame()

    for i in range(nb_iteration):
        print(f"------\nIteration {i+1}")
        dt_random = get_random_date(dt_start, dt_end, seed)
        l_article_url = get_article_list(l_base_url, dt_random)
        df_article_tmp = extract_article(l_article_url)

        df_article_tmp["day"] = dt_random
        df_article = pd.concat([df_article, df_article_tmp], ignore_index=True, sort=False)

    return df_article.reset_index().rename(columns={"index": "id"})


def get_random_date(dt_start: date, dt_end: date, seed: int) -> date:
    """Generate random date between 2 dates

    Args:
        dt_start: start date (included)
        dt_end: end date (excluded)
        seed: random seed to make pipeline reproducible

    Returns: A random date between the 2 boundaries

    """

    time_between_dates = dt_end - dt_start
    delta_end_start = time_between_dates.days
    random.seed(seed)
    random_number_of_days = random.randrange(delta_end_start)

    return dt_start + timedelta(days=random_number_of_days)


def get_article_list(l_base_url: List[str], date: date) -> List[str]:
    """

    Args:
        l_base_url: List of medium blogs to scrap
        date: Archive day

    Returns:
        List of the daily URL articles
    """

    # init variables
    year, month, day = date.strftime('%Y'), date.strftime('%m'), date.strftime('%d')
    l_links = []

    for base_url in l_base_url:

        # Get and parse the medium archive page
        response = requests.get(f"{base_url}/archive/{year}/{month}/{day}", allow_redirects=True)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all articles for the day
        articles = soup.find_all(
            "div",
            class_="streamItem streamItem--postPreview js-streamItem"
        )

        # Update the article URL list
        l_links_tmp = [article.find_all("a")[4]['href'] for article in articles]
        l_links = l_links + l_links_tmp

    return l_links


def extract_article(l_links: List[str]) -> pd.DataFrame:
    """ Extract medium article content from a list of urls

    Args:
        l_links: list of medium url

    Returns:
        Dataframe containing the extracted text
    """

    l_articles = []

    for link in l_links:
        try:
            d_article = {}

            # Parse web page
            data = requests.get(link)
            soup = BeautifulSoup(data.content, 'html.parser')

            # Get title
            title = soup.findAll('title')[0]
            title = title.get_text()
            d_article['title'] = unicodedata.normalize('NFKD', title)
            print(f"Extracting - {title}")

            # Get author
            author = soup.findAll('meta', {"name": "author"})[0]
            author = author.get('content')
            d_article['author'] = unicodedata.normalize('NFKD', author)

            # Set link
            d_article['link'] = link

            # Get page text
            paras = soup.findAll('p')
            text = ''
            nxt_line = '\n'
            for para in paras:
                text += unicodedata.normalize('NFKD', para.get_text()) + nxt_line
            d_article['text'] = text

            # Update list
            l_articles.append(d_article)

        except:
            # for exceptions caused due to change of format on that page
            continue

    return pd.DataFrame(l_articles)

