import pandas as pd
import numpy as np
from urllib.parse import unquote
from bs4 import BeautifulSoup
import os
from tqdm import tqdm
import pickle

def read_articles(file_path='./data/paths-and-graph/articles.tsv'):

    articles = pd.read_csv(file_path, comment='#', names=["article"])
    articles = articles["article"].apply(unquote).replace('_', ' ', regex=True)

    return articles

def read_categories(file_path='./data/paths-and-graph/categories.tsv'):

    # Step 1: Load the data
    categories = pd.read_csv(file_path, sep='\t', comment='#', names=["article", "category"])
    categories["article"] = categories["article"].apply(unquote).replace('_', ' ', regex=True)

    # Step 2: Separate categories by hierarchical levels
    # Find the maximum depth by checking the highest number of splits in any category
    max_depth = categories['category'].str.split('.').map(len).max()

    # Dynamically generate column names based on the max depth
    category_levels = categories['category'].str.split('.', expand=True)
    category_levels.columns = [f'level_{i+1}' for i in range(max_depth)]

    # Concatenate the levels with the original DataFrame
    df_expanded = pd.concat([categories, category_levels], axis=1)

    # Check if level_1 has only one unique value and adjust accordingly, 
    # by removing the column and renaming the rest
    level_1_values = df_expanded['level_1'].unique()
    if len(level_1_values) == 1:
        df_expanded.drop(columns='level_1', inplace=True)
        df_expanded.columns = ['article', 'category'] + [f'level_{i}' for i in range(1, max_depth)]

    return df_expanded

def read_links(file_path='./data/paths-and-graph/links.tsv'):

    links = pd.read_csv(file_path, sep='\t', comment='#', names=["linkSource", "linkTarget"])
    links["linkSource"] = links["linkSource"].apply(unquote).replace('_', ' ', regex=True)
    links["linkTarget"] = links["linkTarget"].apply(unquote).replace('_', ' ', regex=True)

    return links

def read_matrix(file_path='./data/paths-and-graph/shortest-path-distance-matrix.txt'):

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Process each line to convert it into a list of distances
    data = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#'):
            continue  # Skip comment lines and empty lines
        distances = [int(char) if char != '_' else -1 for char in line.strip()]
        data.append(distances)

    matrix = pd.DataFrame(data, dtype=int)

    # Read the articles.tsv file to use as column headers & index
    names_articles = read_articles()
    matrix.columns = names_articles
    matrix.index = names_articles    
    print("The rows are the source articles and the columns are the destination articles")

    return matrix

def read_unfinished_paths(file_path='./data/paths-and-graph/paths_unfinished.tsv'):

    column_names = ['hashedIpAddress', 'timestamp', 'durationInSec', 'path', 'target', "type"]
    df = pd.read_csv(file_path, sep='\t', comment='#', names=column_names)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['path'] = df['path'].apply(unquote).replace('_', ' ', regex=True)


    return df

def read_finished_paths(file_path='./data/paths-and-graph/paths_finished.tsv'):

    column_names = ['hashedIpAddress', 'timestamp', 'durationInSec', 'path', 'rating']
    df = pd.read_csv(file_path, sep='\t', comment='#', names=column_names)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['path'] = df['path'].apply(unquote).replace('_', ' ', regex=True)

    return df

class htmlParser:
    def __init__(self, file_path='./data/articles_html/wp'):
        self.article_URLs = [] # full article paths
        for subdir, dirs, files in os.walk(file_path):
            for file in files:
                article_link = os.path.join(subdir, file).replace('\\', '/')
                if self.is_valid_link(article_link):
                    self.article_URLs.append(article_link)
        self.article_names = self.links_to_articles(self.article_URLs)
        self.parsed_articles = {}

    def parse_all(self):
        '''Parse all the valid articles'''

        for i in tqdm(range(len(self.article_names))):
            self.parsed_articles[self.article_names.iloc[i]] = self.parse_html_article(self.article_URLs[i])

    def parse_selection(self, indices):
        '''
        Parse a selection of valid articles (those with the indices of self.article_names passed)
        For instance, to get all the same articles as in paths-and-graph:
        indices = [i for i in df_html_articles.index if df_html_articles[i] in df_article_names.values]
        '''

        for i in tqdm(indices):
            self.parsed_articles[self.article_names.iloc[i]] = self.parse_html_article(self.article_URLs[i])

    def parse_html_article(self, filename):
        '''
        Parse a wikipedia article: extract word counts and links to other articles.
        This is done for the abstract, h2 categories, h3 subcategories, tables and for the entire article.
        '''
            
        with open(filename, 'r') as html_content:
            try:
                soup = BeautifulSoup(html_content, 'html5lib')
            except:
                return None
            # print(soup.prettify()) # nice overview of the article structure

            # TITLE
            title_tag = soup.find('h1', {'class': 'firstHeading'})
            if hasattr(title_tag, 'get_text'):
                title = title_tag.get_text().strip()
            else: 
                return None # If there is no title it's probably not a wikipedia article

            # BODY
            body_content = soup.find('div', {'id': 'bodyContent'})
            words = body_content.get_text(separator=' ', strip=True).split()
            links = body_content.find_all('a')
            URLs = self.get_URLs_from_links(links)

            # ABSTRACT (short text right under title and before the first h2 category)
            first_h2 = body_content.find('h2')
            abstract_text = ''
            abstract_links = []
            for child in body_content.children:
                if child == first_h2:
                    break
                if hasattr(child, 'get_text'):
                    abstract_text += child.get_text(separator=' ', strip=True) + " "
                if hasattr(child, 'find_all'):
                    abstract_links += child.find_all('a')
            abstract_URLs = self.get_URLs_from_links(abstract_links)
            
            # CATEGORIES AND SUBCATEGORIES
            categories_data = self.parse_categories(body_content)
                    
            # TABLES: special structures, infoboxes, ...
            tables = soup.find_all('table')
            tables_data = []

            for table in tables:
                if table.get('class'):
                    table_class = table.get('class')
                else:
                    table_class = None

                tables_text = table.get_text(separator=' ', strip=True)
                table_links = table.find_all('a')
                table_URLs = self.get_URLs_from_links(table_links)

                tables_data.append({
                    'class': table_class,
                    'num_words': len(tables_text.split()),
                    'table_links': table_URLs
                })

            DATA = {
                'title': title,
                'total_words': len(words),
                'total_links': URLs,
                'abstract_length': len(abstract_text.split()),
                'abstract_links': abstract_URLs,
                'categories_data': categories_data,
                'tables': tables_data
            }

            return DATA
        
    def parse_categories(self, body_content):
        '''
        Parse h2 categories: extract word counts, links to other articles and h3 subcategory information.

        Note: 
        - There is a 'Retrieved from "http://en.wikipedia.org/..."' message of length 4 words 
            at the end of each article, but this is probably negligible
        '''
        categories_data = []

        for h2 in body_content.find_all('h2'):

            category_name = h2.get_text().strip()
            h2_text = ''
            h2_links = []
            
            # h2 categories: for each, find all siblings on the same level
            for sibling in h2.find_next_siblings():
                if sibling.name == 'h2':
                    break
                if hasattr(sibling, 'get_text'):
                    h2_text += sibling.get_text(separator=' ', strip=True) + '\n'
                if hasattr(sibling, 'find_all'):
                    h2_links += sibling.find_all('a')

            h2_URLs = self.get_URLs_from_links(h2_links)
                
            categories_data.append({
                'name': category_name,
                'num_words': len(h2_text.split()),
                'h2_links': h2_URLs,
                'subcategories': self.parse_subcategories(h2) # h3 categories: for each h2 category, check if there are subcategories
            })

        return categories_data

    def parse_subcategories(self, h2):
        '''Parse h3 categories: extract word counts, links to other articles from the given h2 category.'''

        subcategories_data = []

        for tag in h2.find_all_next():

            # if h3, parse it. If h2, we are in the next section. If other, continue searching-
            if tag.name == 'h3':
                h3 = tag
            elif tag.name == 'h2':
                break
            else: 
                continue
            
            subcategory_name = h3.get_text().strip()
            h3_text = ''
            h3_links = []

            for sibling in h3.find_next_siblings():
                if sibling.name == 'h3':
                    break
                if hasattr(sibling, 'get_text'):
                    h3_text += sibling.get_text(separator=' ', strip=True) + '\n'
                if hasattr(sibling, 'find_all'):
                    h3_links += sibling.find_all('a')

                h3_URLs = self.get_URLs_from_links(h3_links)

            subcategories_data.append({
                'name': subcategory_name,
                'num_words': len(h3_text.split()),
                'h3_links': h3_URLs
            })

        return subcategories_data
    
    def get_overview(self, parsed_data):

        print("Page Overview")
        print("-------------")
        print(f"Title: {parsed_data['title']}")
        print(f"Total Words: {parsed_data['total_words']}")
        print(f"Total Links: {len(parsed_data['total_links'])}\n")
        
        # Abstract Summary
        print("Abstract Overview")
        print("-----------------")
        print(f"Abstract Length (words): {parsed_data['abstract_length']}")
        print(f"Abstract Links: {len(parsed_data['abstract_links'])}\n")
        
        # Categories and Subcategories Summary
        print("Categories and Subcategories Overview")
        print("-------------------------------------")
        for i, category in enumerate(parsed_data['categories_data'], start=1):
            print(f"Category {i}: {category['name']}")
            print(f" - Words: {category['num_words']}")
            print(f" - Links: {len(category['h2_links'])}")
            if 'subcategories' in category:
                for j, subcategory in enumerate(category['subcategories'], start=1):
                    print(f"   Subcategory {j}: {subcategory['name']}")
                    print(f"   - Words: {subcategory['num_words']}")
                    print(f"   - Links: {len(subcategory['h3_links'])}")
            print()
        
        # Tables Summary
        print("Tables Overview")
        print("---------------")
        for i, table in enumerate(parsed_data['tables'], start=1):
            print(f"Table {i}:")
            print(f" - Class(es): {table['class']}")
            print(f" - Words: {table['num_words']}")
            print(f" - Links: {len(table['table_links'])}")
        
    def get_URLs_from_links(self, links):
        '''Get the valid URLs from soup links'''

        link_href = [link.get('href') for link in links if link.get('href')]
        link_urls = [link for link in link_href if self.is_valid_link(link)]

        return link_urls

    def is_valid_link(self, link):
        '''Checks if a link redirects to another valid article'''

        if 'wp' in link and link.endswith('.htm') and 'wp/index/' not in link:
            return True

        return False
        
    def links_to_articles(self, links):
        '''Transforms a list of links to a list of decoded articles. Requires links to be valid. '''

        # First splits to eliminate the root directory, then removes the file type. Indexing with [:2] to remove the one-letter directory /*
        article_names = [link.split('wp/')[1].split('.htm')[0][2:] for link in links]

        # unquote twice because there are some double encodings (e.g. Georgia_%2528country%2529)
        df_article_names = pd.Series(article_names, name='article').replace('_', ' ', regex=True).apply(unquote).apply(unquote) 

        return df_article_names
    
    def save_pickle(self, filename='./data/paths-and-graph/parsed_html.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump(self.parsed_articles, file)

    def load_pickle(self, filename='./data/paths-and-graph/parsed_html.pkl'):
        with open(filename, 'rb') as file:
            self.parsed_articles = pickle.load(file)