import pandas as pd
import numpy as np
from urllib.parse import unquote
from bs4 import BeautifulSoup
import os
from tqdm import tqdm
import pickle


def get_bad_articles():
    '''Returns the name and index of the articles in articles.tsv that are not wikipedia articles or that have missing propreties (e.g. missing categories) and should be removed.'''

    filepath='./data/paths-and-graph/articles.tsv'
    articles = pd.read_csv(filepath, comment='#', names=["article"])
    articles = articles["article"].apply(unquote).replace('_', ' ', regex=True)

    # The bad articles are not real wikipedia pages, cannot be parsed by the html parser and are do not appear in user paths
    bad_articles = ['Directdebit', 'Donation', 'Friend Directdebit', 'Sponsorship Directdebit', 'Wowpurchase', 'Pikachu','Wikipedia Text of the GNU Free Documentation License']
    bad_articles_idx = articles[articles.isin(bad_articles)].index

    return bad_articles, bad_articles_idx.to_list()

def read_articles():
    '''
    Return a Series with the article names, with '_' removed and the percent encoding unquoted.
    Removes the articles in the list that are not wikipedia articles.
    '''

    filepath='./data/paths-and-graph/articles.tsv'
    articles = pd.read_csv(filepath, comment='#', names=["article"])
    articles = articles["article"].apply(unquote).replace('_', ' ', regex=True)
    
    # Remove invalid articles
    bad_articles, _ = get_bad_articles()
    articles = articles[~articles.isin(bad_articles)].reset_index(drop=True)

    return articles

def read_categories():

    # Step 1: Load the data
    filepath='./data/paths-and-graph/categories.tsv'
    categories = pd.read_csv(filepath, sep='\t', comment='#', names=["article", "category"])
    categories["article"] = categories["article"].apply(unquote).replace('_', ' ', regex=True)
    categories["category"] = categories["category"].apply(unquote).replace('_', ' ', regex=True)

    # Remove invalid articles
    categories = categories.loc[categories['article'].isin(read_articles()), :].reset_index(drop=True)

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

def read_links():
    '''Finds all the existing links between articles, removes invalid articles'''

    filepath='./data/paths-and-graph/links.tsv'
    links = pd.read_csv(filepath, sep='\t', comment='#', names=["linkSource", "linkTarget"])
    links["linkSource"] = links["linkSource"].apply(unquote).replace('_', ' ', regex=True)
    links["linkTarget"] = links["linkTarget"].apply(unquote).replace('_', ' ', regex=True)
    
    # Remove invalid articles
    links = links.loc[links['linkSource'].isin(read_articles()), :].reset_index(drop=True)
    links = links.loc[links['linkTarget'].isin(read_articles()), :].reset_index(drop=True)

    return links

def read_shortest_path_matrix():
    '''The rows are the source articles and the columns are the destination articles'''
    
    filepath='./data/paths-and-graph/shortest-path-distance-matrix.txt'
    with open(filepath, 'r') as file:
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

    # Drop bad articles
    _, bad_articles_idx = get_bad_articles()
    matrix.drop(index=bad_articles_idx, inplace=True)
    matrix.drop(columns=bad_articles_idx, inplace=True)

    # Read the articles.tsv file to use as column headers & index
    names_articles = read_articles()
    matrix.columns = names_articles
    matrix.index = names_articles  

    return matrix

def read_unfinished_paths():

    filepath='./data/paths-and-graph/paths_unfinished.tsv'
    column_names = ['hashedIpAddress', 'timestamp', 'durationInSec', 'path', 'target', 'type']
    df = pd.read_csv(filepath, sep='\t', comment='#', names=column_names)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['path'] = df['path'].apply(unquote).replace('_', ' ', regex=True)
    df['target'] = df['target'].apply(unquote).replace('_', ' ', regex=True)

    print("Unfinished Paths \nNumber of rows before filtering:", len(df))
    # Drop invalid articles
    valid_articles = set(read_articles())
    valid_articles.add("<")

    invalid_target_articles = set()
    for target in df['target']:
        if target not in valid_articles:
            invalid_target_articles.add(target)
    print("Invalid target articles found:", invalid_target_articles)
    # Drop rows with invalid target articles
    df = df.loc[df['target'].isin(valid_articles)].reset_index(drop=True)
    
    invalid_articles_set = set()
    # Filter and find invalid articles
    def check_path(path):
        articles = path.split(';')
        for article in articles:
            article = article.strip()
            if article not in valid_articles:
                invalid_articles_set.add(article)  # Add only the invalid article
                return False  # Exclude this row if any invalid article is found
        return True  # Include this row if all articles are valid

    # Apply the filter with the custom function
    df = df[df["path"].apply(check_path)].reset_index(drop=True)

    # Print unique invalid articles
    print("Invalid articles found in path:", invalid_articles_set)
    print("Number of rows after filtering:", len(df),"\n")

    return df

def read_finished_paths():

    filepath='./data/paths-and-graph/paths_finished.tsv'
    column_names = ['hashedIpAddress', 'timestamp', 'durationInSec', 'path', 'rating']
    df = pd.read_csv(filepath, sep='\t', comment='#', names=column_names)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['path'] = df['path'].apply(unquote).replace('_', ' ', regex=True)

    print("Finished Paths \nNumber of rows before filtering:", len(df))
    # Drop invalid articles
    valid_articles = set(read_articles())
    valid_articles.add("<")
    
    # Set to store unique invalid articles
    invalid_articles_set = set()

    # Filter and find invalid articles
    def check_path(path):
        articles = path.split(';')
        for article in articles:
            article = article.strip()
            if article not in valid_articles:
                invalid_articles_set.add(article)  # Add only the invalid article
                return False  # Exclude this row if any invalid article is found
        return True  # Include this row if all articles are valid

    # Apply the filter with the custom function
    df = df[df["path"].apply(check_path)].reset_index(drop=True)

    # Print unique invalid articles
    print("Invalid articles found in path:", invalid_articles_set)
    print("Number of rows after filtering:", len(df),"\n")
    
    return df

def read_similartiy_matrix():
    filepath='./data/paths-and-graph/similarity_matrix.npy'
    article_names = read_articles()
    sm = np.load(filepath)
    df_sm = pd.DataFrame(sm)
    df_sm.columns = article_names
    df_sm.index = article_names

    return df_sm

def find_shortest_distance(row, distance_matrix):
    '''
    Finds the start and the target of a path and returns the shortest distance between the two.
    Distance can be anything: shortest path, semantic cosine similarity, ...
    '''
    
    articles = row['path'].split(';')
    if 'target' in row.index:
        return distance_matrix.loc[articles[0]][row['target']]
    return distance_matrix.loc[articles[0]][articles[-1]]

def replace_back_clicks(path):
    '''Replaces back clicks < with the article that was landed on'''
    articles = path.split(';')
    resolved_path = []
    consecutive_backclicks = 0
    for i, art in enumerate(articles):
        if art == '<':
            resolved_path.append(resolved_path[i-2-consecutive_backclicks])
            consecutive_backclicks += 2
        else:
            consecutive_backclicks=0
            resolved_path.append(art)
    
    return resolved_path

class htmlParser:
    def __init__(self, filepath='./data/articles_html/wp'):
        self.article_URLs = [] # full article paths
        for subdir, dirs, files in os.walk(filepath):
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
            linked_articles = self.links_to_articles(URLs)

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
            abstract_linked_articles = self.links_to_articles(abstract_URLs)
            
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
                table_linked_articles = self.links_to_articles(table_URLs)

                tables_data.append({
                    'class': table_class,
                    'num_words': len(tables_text.split()),
                    'table_links': table_linked_articles
                })

            DATA = {
                'title': title,
                'total_words': len(words),
                'total_links': linked_articles,
                'abstract_length': len(abstract_text.split()),
                'abstract_links': abstract_linked_articles,
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
            h2_linked_articles = self.links_to_articles(h2_URLs)
                
            categories_data.append({
                'name': category_name,
                'num_words': len(h2_text.split()),
                'h2_links': h2_linked_articles,
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
                h3_linked_articles = self.links_to_articles(h3_URLs)

            subcategories_data.append({
                'name': subcategory_name,
                'num_words': len(h3_text.split()),
                'h3_links': h3_linked_articles
            })

        return subcategories_data
    
    def get_overview(self, parsed_data):
        '''Print out the overview of an article: abstract words and links, total word and links at statistics for (sub)categories and tables'''

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
    
    def save_pickle(self):
        filename='./data/paths-and-graph/parsed_html.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self.parsed_articles, file)

    def load_pickle(self):
        filename='./data/paths-and-graph/parsed_html.pkl'
        with open(filename, 'rb') as file:
            self.parsed_articles = pickle.load(file)
    
    def get_df_html_stats(self):
        '''
        Get a Dataframe with general information over the parsed data,
        namely word and link counts, link density, number of categories and special elements.
        '''

        articles_stats = []
        for key, art in self.parsed_articles.items():
            if art is None: 
                continue
            article_stats = {
                'article_name': art['title'],
                'total_words': art['total_words'],
                'total_links': len(art['total_links']),
                'link_density': len(art['total_links']) / art['total_words'],
                'abstract_words': art['abstract_length'],
                'abstract_links': len(art['abstract_links']),
                'abstract_link_density': len(art['abstract_links']) / art['abstract_length'],
                'num_categories': len(art['categories_data']),
                'num_subcategories': sum(len(cat.get('subcategories', [])) for cat in art['categories_data']),
                'num_tables': len(art['tables'])
            }
            articles_stats.append(article_stats)

        df_html_stats = pd.DataFrame(articles_stats)

        return df_html_stats
    
    def find_link_positions(self, article_start, article_next):
        '''
        Finds where the link to article_next is placed inside article_start. 
        Returns in which category/table it is placed in, and how many links and words roughly preceded it.
        If a link is present multiple times, returns all the instances found.
        '''
        link_position = {}

        # global position in article
        article_start = self.parsed_articles[article_start]
        occurences = article_start['total_links'][article_start['total_links'] == article_next].index.tolist()
        link_position['total_links'] = len(article_start['total_links'])
        link_position['total_words'] = article_start['total_words']
        link_position['article_link_position'] = [x+1 for x in occurences]

        # occurences in abstract
        occurences_abstract = article_start['abstract_links'][article_start['abstract_links'] == article_next].index.tolist()
        if occurences_abstract:
            link_position['abstract_links'] = len(article_start['abstract_links'])
            link_position['abstract_link_position'] = [x+1 for x in occurences_abstract]

        # occurences in subcategories
        occurences_categories = []
        words_prior = article_start['abstract_length'] 
        for category in article_start['categories_data']:
            occurences_category = category['h2_links'][category['h2_links'] == article_next].index.tolist()
            words_prior += category['num_words']
            if occurences_category:
                occurences_categories.append(occurences_category)
        '''occurences_subcategories = article_start['abstract_links'][article_start['abstract_links'] == article_next].index
        occurences_tables = article_start['abstract_links'][article_start['abstract_links'] == article_next].index'''

        # occurences in tables

        return link_position
    
    def find_path_link_positions(self, df_paths):
        '''
        For every path of user games, finds where on the wikipedia page the link could have been clicked
        df_finished.sample(1, random_state=2)
        '''

        for path in df_paths.sample(1, random_state=2)['path']:
            articles = path.split(';')
            no_back_clicks = replace_back_clicks(path)
            print(articles)
            print(no_back_clicks)
            
            for a in range(len(articles)-1):
                if articles[a+1] == '<':
                    continue
                else:
                    print(f'{no_back_clicks[a]}: {self.find_link_positions(no_back_clicks[a], no_back_clicks[a+1])}')
