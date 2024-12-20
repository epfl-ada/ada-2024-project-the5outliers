import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import Counter
import config

def hex_to_rgba(hex_color):
    # Remove the hash symbol if present
    hex_color = hex_color.lstrip('#')

    return f'rgba({int(hex_color[0:2], 16)},{int(hex_color[2:4], 16)},{int(hex_color[4:6], 16)},{0.2})'

def compute_steady_state(markov_transitions, df_article_names, backclicks=False):
    """
    Compute the steady state distribution of a Markov chain.
    Parameters:
    markov_transitions (numpy.ndarray): A square matrix representing the transition probabilities of the Markov chain.
    df_article_names (pandas.Series): A series containing the names of the articles corresponding to the states of the Markov chain.
    backclicks (bool, optional): If True, includes a backclick state represented by '<' in the article names. Default is False.
    Returns:
    pandas.DataFrame: A DataFrame with two columns:
        - 'articles': The names of the articles (states).
        - 'steady_state_proportion': The steady state distribution of the Markov chain.
    """


    val, vec = np.linalg.eig(markov_transitions.T)
    ss = np.real(-vec[:, 0]) / np.real(-vec[:, 0]).sum()

    if backclicks:
        df_article_names = pd.concat([pd.Series('<'), df_article_names]).reset_index(drop=True)

    steady_state =pd.DataFrame()
    steady_state['articles'] = df_article_names
    steady_state['steady_state_proportion'] = ss

    return steady_state

def get_nth_transition_matrix(df, article_names, transition, backclicks=False, normalise=True):
    """
    Create the nth transition matrix based on the user paths.

    Parameters:
    - df (pd.DataFrame): The finished or unfinished paths.
    - article_names (pd.Series): The names of all the articles to consider for transitions
    - n (int): The position in the path to compute the transition (1 for first, 2 for second, etc.).

    Returns:
    - pd.DataFrame: A transition matrix for the nth position, with article_names as both rows and columns. Missing transitions are filled with 0.
    """

    if backclicks:
        article_names = pd.concat([pd.Series(['<'], name='article'), article_names], names='article').reset_index(drop=True)
    
    df = df.copy()
    df['path_split'] = df['path'].str.split(';')
    df['current'] = df['path_split'].apply(lambda x: x[transition-1] if len(x) >= transition else None)
    df = df.dropna(subset=['current'])
    
    if df['current'].nunique() < 2:
        return pd.DataFrame(0, index=article_names, columns=article_names)
    
    df['next'] = df['path_split'].apply(lambda x: x[transition] if len(x) > transition else None)
    transition_counts = df.groupby(['current', 'next']).size().reset_index(name='count')

    transition_matrix = transition_counts.pivot(index='current', columns='next', values='count').fillna(0)
    transition_matrix = transition_matrix.reindex(index=article_names, columns=article_names, fill_value=0)

    if normalise:
        transition_matrix = row_normalise(transition_matrix)
    
    return transition_matrix

def row_normalise(matrix):
    """Row-normalise the dataframe given in parameters"""

    row_sums = matrix.sum(axis=1)  
    normalized_matrix = matrix.div(row_sums, axis=0)  
    normalized_matrix[row_sums == 0] = 0  # Keep rows with sum = 0 as zero

    return normalized_matrix

def get_transition_probabilities(df_article_names, parser, backclicks=False, normalise=True):
    """
    Compute the transition probabilities for navigating between articles.

    Parameters:
    - df_article_names (pd.Series): A series of article names, representing the list of articles.
    - parser (object): An object with a `parsed_articles` attribute, where `parsed_articles` is a dictionary 
    that maps article names to their associated metadata, including a 'total_links' key containing the links from the article.
    - backclicks (bool, optional): If True, includes a "backclick" action as a possible navigation option,
    where backclick allows transitioning equally to any other article. Defaults to False.

    Returns:
    - np.ndarray: A square matrix of shape (actions, actions), where `actions` is the number of articles (or articles + 1 if backclicks are enabled). 
    Each entry [i, j] represents the probability of transitioning from article `i` to article `j`.
    """


    # are there any articles with no links? Not anymore. Are there disjoint sets that you cannot travel in both directions? I don't know
    actions = len(df_article_names) 

    if backclicks: 
        actions += 1

    transition_probabilities = np.zeros((actions, actions))
    
    if backclicks:
        transition_probabilities[:, 0] = 1  # in each article, backclick is a unique possible click choice
        transition_probabilities[0, :] = 1  # the article "backclick" is equally likely to lead to any article

    for i, art in enumerate(df_article_names):
        art_links = pd.Series(parser.parsed_articles[art]['total_links']).value_counts()
        indx = df_article_names[df_article_names.isin(art_links.index)]
        arts = np.array(indx.index.tolist())
        vals = art_links.loc[indx].values

        if backclicks: 
            i += 1
            arts += 1

        if arts.size > 0:
            transition_probabilities[i, arts] += vals

    if normalise:
        transition_probabilities /= transition_probabilities.sum(axis=1)[:, np.newaxis]

    return transition_probabilities

def get_step_divergences(df_article_names, parser, df_paths, num_steps=10, df_categories=None, backclicks=False):
    """
    Calculate the Kullback-Leibler (KL) divergence between user transition matrices and Markov transition matrices 
    for a given number of steps and return the mean and standard deviation of the KL divergence for each step.
    
    Parameters:
    -----------
        df_article_names (pd.DataFrame): DataFrame containing article names.
        parser (object): Parser object to process the articles.
        df_paths (pd.DataFrame): DataFrame containing user navigation paths.
        num_steps (int, optional): Number of steps to calculate the transition matrices for. Default is 10.
    Returns:
    --------
        pd.DataFrame: DataFrame with MultiIndex columns containing the mean and standard deviation of the KL divergence 
                    for each step. The rows are sorted by the mean KL divergence of the first step in descending order.
    """
    
    KL_stats_step = []   # List to store mean and std of KL divergence for each step.
    markov_transitions = get_transition_probabilities(df_article_names, parser, backclicks=backclicks)

    for n in range(1, num_steps + 1):
        # Get transition matrices for the nth step for both users and Markov.
        user_transitions_n = get_nth_transition_matrix(df_paths, df_article_names, n, backclicks=backclicks)

        # Compute KL divergence for each transition.
        KL_n = np.where((user_transitions_n > 0) & (markov_transitions > 0),
                        user_transitions_n * np.log(user_transitions_n / markov_transitions), 0)
        KL_n_df = pd.DataFrame(KL_n, columns=user_transitions_n.columns, index=user_transitions_n.index)
        
        if df_categories is not None:
            if backclicks:
                df_categories = df_categories[['article', 'level_1']]
                df_categories = pd.concat([pd.DataFrame([['<', '<']], columns=['article', 'level_1']), df_categories], ignore_index=True)
            
            # Inverse rows and columns and get article as a column
            KL_cat_n_df = KL_n_df.T.reset_index(names='article')
            KL_cat_n_df = KL_cat_n_df.merge(right=df_categories[['article', 'level_1']], on='article', how='left')
            KL_cat_n_df = KL_cat_n_df.drop(columns='article')

            group_means = pd.DataFrame()
            group_means['mean'] = KL_cat_n_df.groupby('level_1').apply(lambda group: group.iloc[:, :-1].values.mean())
            group_means['sem'] = 1.96 * KL_cat_n_df.groupby('level_1').apply(lambda group: group.iloc[:, :-1].values.std() / np.sqrt(group.iloc[:, :-1].size))

            KL_stats_step.append(group_means)
        else:
            # Compute mean and standard deviation for KL divergence at this step.
            KL_stats_step.append({
                "mean": KL_n_df.mean(axis=0),
                "sem": 1.96 * KL_n_df.sem(axis=0)
            })

    # Convert KL stats to a DataFrame with MultiIndex columns (mean and std).
    KL_stats_step = pd.concat([pd.DataFrame(stats) for stats in KL_stats_step], keys=range(1, num_steps + 1), axis=1)
    
    # Flatten MultiIndex for clarity (step indices and metrics in single-level columns).
    KL_stats_step.columns = pd.MultiIndex.from_tuples(
        [(step, metric) for step, metric in KL_stats_step.columns]
    )
    
    # Sort rows by the mean of the first step.
    KL_stats_step = KL_stats_step.sort_values(by=(1, "mean"), ascending=False)

    return KL_stats_step

def plot_article_step_divergence(step_divergence, df_categories, N_articles):
    """
    Plots the stepwise divergence (mean and std) from a random path for different articles.
    Parameters:
    step_divergence (pd.DataFrame): A DataFrame where each column represents the mean and std divergence
                                    for an article at each step, and the index represents the articles.
    color_dict (dict): A dictionary mapping article names to specific colors for plotting.
    Returns:
    None: This function displays a plot using Plotly Express.
    """

    PALETTE_ARTICLE_DICT_COLORS = {}
    for article in step_divergence.index:
        art_row = df_categories[df_categories['article']==article]
        PALETTE_ARTICLE_DICT_COLORS[art_row['article'].item()] = config.PALETTE_CATEGORY_DICT_COLORS.get(art_row['level_1'].item())

    PALETTE_ARTICLE_DICT_COLORS['<'] = '#000000'

    # Flatten multi-index
    df_mean = step_divergence.xs('mean', level=1, axis=1)
    df_sem = step_divergence.xs('sem', level=1, axis=1)
    df_long_mean = df_mean.reset_index().melt(id_vars='article', var_name='Step', value_name='Divergence')
    df_sem_long = df_sem.reset_index().melt(id_vars='article', var_name='Step', value_name='Error')
    df_combined = pd.merge(df_long_mean, df_sem_long, on=['article', 'Step'])

    fig = go.Figure()
    '''
    fig = px.line(df_combined, x='Step', y='Divergence', color='article', markers=True, 
                    color_discrete_map=color_dict, error_y='Error')
    '''

    # Add horizontal line for random path
    fig.add_hline(y=0, line=dict(color='black', dash='dash'),
                annotation_text="Random Path", annotation_position="bottom right",
                annotation_font=dict(size=12, color="black"))

    for article in df_combined['article'].unique():
        article_data = df_combined[df_combined['article'] == article].reset_index(drop=True)
        x = article_data['Step'].to_list()
        y = article_data['Divergence']
        upper_bound = y + article_data['Error']
        lower_bound = y - article_data['Error']
        legend = f"{article} ({df_categories[df_categories['article'] == article]['level_1'].item()})"

        # Line trace
        fig.add_trace(go.Scatter(
            x=x, y=y, 
            name=legend,
            line_color=PALETTE_ARTICLE_DICT_COLORS[article],
            legendgroup=legend,  # Group traces by article name
            showlegend=True
        ))

        # Error region trace
        fig.add_trace(go.Scatter(
            x=x + x[::-1],  # x, then reversed x
            y=upper_bound.to_list() + lower_bound.to_list()[::-1],  # upper, then lower reversed
            fill='toself',
            fillcolor=hex_to_rgba(PALETTE_ARTICLE_DICT_COLORS[article]),  # Convert color for transparency
            line=dict(color='rgba(255,255,255,0)'),  # Invisible border line
            hoverinfo="skip",  # Skip hover info for the region
            showlegend=False,  # Do not show a separate legend entry
            legendgroup=legend  # Group with the corresponding line trace
        ))

    fig.update_layout(
        xaxis=dict(
            range=[df_long_mean['Step'].min() - 0.5, df_long_mean['Step'].max() + 0.5],
            title='Step',
            tickmode='linear'  # Ensures all integer ticks are shown
        ),
        yaxis=dict(title='Divergence Value'),
        width=900,  # Set the width of the plot
        height=600, # Set the height of the plot
        title=dict(text=f"Mean Stepwise Divergence from Random Path for {N_articles} Articles with Highest Divergence"),
        legend_tracegroupgap=3,
        paper_bgcolor="#fafaf9", 
        plot_bgcolor="#fff",  

    )

    return fig

def plot_category_step_divergence(step_divergence, color_dict):
    """
    Plots the stepwise deviation from a random path for different categories.
    Parameters:
    step_divergence (pd.DataFrame): DataFrame containing the stepwise divergence values for each article.
    df_categories (pd.DataFrame): DataFrame containing the article categories with at least 'article' and 'level_1' columns.
    color_dict (dict): Dictionary mapping category names to colors for the plot.
    Returns:
    None: Displays an interactive plotly line plot with markers.
    """
    

    # Flatten multi-index
    df_mean = step_divergence.xs('mean', level=1, axis=1)
    df_sem = step_divergence.xs('sem', level=1, axis=1)
    df_long_mean = df_mean.reset_index().melt(id_vars='level_1', var_name='Step', value_name='Divergence')
    df_sem_long = df_sem.reset_index().melt(id_vars='level_1', var_name='Step', value_name='Error')
    df_combined = pd.merge(df_long_mean, df_sem_long, on=['level_1', 'Step'])

    fig = go.Figure()

    # Add horizontal line for random path
    fig.add_hline(y=0, line=dict(color='black', dash='dash'),
                annotation_text="Random Path", annotation_position="bottom right",
                annotation_font=dict(size=12, color="black"))

    for cat in df_combined['level_1'].unique():

        article_data = df_combined[df_combined['level_1'] == cat].reset_index(drop=True)
        x = article_data['Step'].to_list()
        y = article_data['Divergence']
        upper_bound = y + article_data['Error']
        lower_bound = y - article_data['Error']

        fig.add_trace(go.Scatter(
            x=x, y=y, name=cat,
            line_color=color_dict[cat],
            legendgroup=cat
        ))

        fig.add_traces(go.Scatter(
            x=x+x[::-1], # x, then x reversed
            y=upper_bound.to_list()+lower_bound.to_list()[::-1], # upper, then lower reversed
            fill='toself',
            fillcolor=hex_to_rgba(color_dict[cat]),
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=cat
        ))

    fig.update_layout(
        xaxis=dict(
            range=[df_long_mean['Step'].min() - 0.5, df_long_mean['Step'].max() + 0.5],
            title='Step',
            tickmode='linear'  # Ensures all integer ticks are shown
        ),
        yaxis=dict(title='Divergence Value'),
        width=800,  # Set the width of the plot
        height=600, # Set the height of the plot
        title=dict(text="Mean Stepwise Divergence from Random Path per Category"),
        legend_tracegroupgap=3,
        paper_bgcolor="#fafaf9", 
        plot_bgcolor="#fff",  
    )

    return fig

def markov_example(parser, user_transitions):
    """
    Generates and visualizes Markov transition probabilities and compares them with user transition data.
    Parameters:
    parser (object): An object used to parse the transition probabilities.
    user_transitions (pd.DataFrame): A DataFrame containing user transition data.
    This function performs the following steps:
    1. Generates transition probabilities for a set of example articles using the provided parser.
    2. Visualizes the transition probabilities as a heatmap.
    3. Computes the steady-state distribution of the transition matrix.
    4. Compares the Markov transition probabilities with user transition data for a subset of articles.
    5. Visualizes the Markov transition probabilities, user transition probabilities, and their differences as heatmaps.
    The function displays the heatmaps using matplotlib and seaborn.
    """


    example_articles = pd.Series(['United States', 'France', 'Agriculture', 'Mexico', 'Natural gas'])

    tp = get_transition_probabilities(example_articles, parser, backclicks=True)

    plt.figure(figsize=(8, 6))
    sn.heatmap(np.linalg.matrix_power(tp, 1), cmap='BuPu', annot=True, fmt=".2f", cbar=True)

    plt.xticks(np.arange(6) + 0.5, ['<'] + example_articles.to_list(), rotation=0)  # Keep x-ticks horizontal
    plt.yticks(np.arange(6) + 0.5, ['<'] + example_articles.to_list(), rotation=0)  # Make y-ticks horizontal

    plt.tight_layout()
    plt.show()

    eigval, eigvec = np.linalg.eig(tp.T)
    steady_state = eigvec[:, 0] / np.sum(eigvec[:, 0])
    assert np.isclose(steady_state @ tp, steady_state) # testing if the vector is really a left eigenvector with eigenvalue 1

    example_articles = pd.Series(['United States', 'France', 'Agriculture', 'Mexico'])
    markov_example = get_transition_probabilities(example_articles, parser, backclicks=False)
    user_example = row_normalise(user_transitions[example_articles].loc[example_articles])
    diff_example = user_example-markov_example

    plt.figure(figsize=(15, 4))

    plt.subplot(131)
    sn.heatmap(markov_example, cmap='BuPu', annot=True, fmt=".2f", cbar=True, vmin=0, vmax=1)
    plt.xticks(np.arange(4) + 0.5, example_articles.to_list(), rotation=0)  # Keep x-ticks horizontal
    plt.yticks(np.arange(4) + 0.5, example_articles.to_list(), rotation=0)  # Make y-ticks horizontal
    plt.title('Markov Transition')

    plt.subplot(132)
    sn.heatmap(user_example, cmap='BuPu', annot=True, fmt=".2f", cbar=True, vmin=0, vmax=1)
    plt.xticks(np.arange(4) + 0.5, example_articles.to_list(), rotation=0)  # Keep x-ticks horizontal
    plt.yticks(np.arange(4) + 0.5, example_articles.to_list(), rotation=0)  # Make y-ticks horizontal
    plt.title('User Transition')

    plt.subplot(133)
    sn.heatmap(diff_example, cmap='vlag', annot=True, fmt=".2f", cbar=True, vmin=-1, vmax=1)
    plt.xticks(np.arange(4) + 0.5, example_articles.to_list(), rotation=0)  # Keep x-ticks horizontal
    plt.yticks(np.arange(4) + 0.5, example_articles.to_list(), rotation=0)  # Make y-ticks horizontal
    plt.title('Difference')

    plt.tight_layout()
    plt.show()

def plot_transitions_normalized(df_article_names, parser, df_categories, palette):
    """
    Visualizes the normalized ratio of transitions to article counts for each category.

    This function computes the Markov transition probabilities for a set of articles, 
    maps them to their respective categories, and calculates the ratio of total transitions 
    to the number of articles in each category. The results are plotted as a bar chart.

    Parameters:
    -----------
    df_article_names : list or pd.Series
        A list or Series of article names for which transitions are computed.

    parser : object
        A parser object used to compute Markov transition probabilities. It is assumed 
        to have the necessary methods or configurations required by `get_transition_probabilities`.

    df_categories : pd.DataFrame
        A DataFrame mapping articles to their categories. It should contain:
        - 'article': Column with article names.
        - 'level_1': Column with corresponding category labels.

    palette : dict
        A dictionary mapping category names to specific colors for the plot.
    """
    markov_transitions = get_transition_probabilities(df_article_names, parser, backclicks=False, normalise=False)
    article_to_category = dict(zip(df_categories['article'], df_categories['level_1']))
    mapped_categories = [article_to_category.get(article, 'Unknown') for article in df_article_names]
    transitions_per_category = pd.DataFrame({'category': mapped_categories, 'transition': markov_transitions.sum(axis=1)})
    transitions_per_category=transitions_per_category.groupby('category', as_index=False).sum()
    category_counts = Counter(mapped_categories)
    category_counts = pd.DataFrame(category_counts.items(), columns=['category', 'count'])

    transitions_per_category = pd.merge(category_counts , transitions_per_category, on='category')
    transitions_per_category['ratio'] =  transitions_per_category['transition'] / transitions_per_category['count']
    transitions_per_category = transitions_per_category.sort_values(by='ratio')

    sn.barplot(data=transitions_per_category, x='category', y='ratio', hue='category', legend=None, palette=palette)

    # Customize the plot
    plt.xlabel('Category')
    plt.ylabel('Transitions / # Articles in Category')
    plt.title('Ratio of Count to Transitions by Category')
    plt.xticks(rotation=90) 
    plt.show()