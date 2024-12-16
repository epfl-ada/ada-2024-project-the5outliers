import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import plotly.express as px


def get_nth_transition_matrix(df, article_names, transition, normalise=True):
    """
    Create the nth transition matrix based on the user paths.

    Parameters:
    - df (pd.DataFrame): The finished or unfinished paths.
    - article_names (pd.Series): The names of all the articles to consider for transitions
    - n (int): The position in the path to compute the transition (1 for first, 2 for second, etc.).

    Returns:
    - pd.DataFrame: A transition matrix for the nth position, with article_names as both rows and columns. Missing transitions are filled with 0.
    """
    
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

def get_step_divergences(df_article_names, parser, df_paths, num_steps=10):
    """
    Calculate the step-wise divergences between user transitions and Markov chain transitions.
    Parameters:
    df_article_names (pd.DataFrame): DataFrame containing article names.
    parser (object): Parser object to parse the articles.
    df_paths (pd.DataFrame): DataFrame containing user paths.
    num_steps (int, optional): Number of steps to consider for the transition matrices. Default is 10.
    Returns:
    tuple: A tuple containing two DataFrames:
        - mean_diff_step (pd.DataFrame): DataFrame of the mean differences between user and Markov transitions for each step.
        - mean_KL_step (pd.DataFrame): DataFrame of the mean Kullback-Leibler (KL) divergences for each step.
    """
    

    mean_diff_step = [] #list of the difference between game and random : represents user choice 
    mean_KL_step = []
    markov_transitions = get_transition_probabilities(df_article_names, parser, backclicks=False)

    for n in range(1, num_steps+1):
        # get transition matrices of the 5 first steps of users and markov
        user_transitions_n = get_nth_transition_matrix(df_paths, df_article_names, n)
        
        # compute column wise sum of the difference between user and random : gives the sum of probalities of player voluntarily chosing to transition to article j
        diff_n = user_transitions_n - markov_transitions
        mean_diff_n = diff_n.sum(axis=0).sort_values(ascending=False)
        mean_diff_step.append(mean_diff_n)

        # compute KL
        KL_n = np.where((user_transitions_n > 0) & (markov_transitions > 0), user_transitions_n * np.log(user_transitions_n / markov_transitions), 0)
        KL_n_df = pd.DataFrame(KL_n, columns=user_transitions_n.columns, index=user_transitions_n.index)
        mean_KL_n = KL_n_df.sum(axis=0).sort_values(ascending=False)
        mean_KL_step.append(mean_KL_n)

    return pd.DataFrame(mean_diff_step), pd.DataFrame(mean_KL_step)

def plot_article_step_divergence(step_divergence, color_dict):
    """
    Plots the stepwise divergence from a random path for different articles.
    Parameters:
    step_divergence (pd.DataFrame): A DataFrame where each column represents the cumulative divergence 
                                    of an article at each step, and the index represents the steps.
    Returns:
    None: This function displays a plot using Plotly Express.
    """

    df_long = step_divergence.reset_index().melt(id_vars='index', var_name='Article', value_name='Cumulative Divergence')
    df_long.rename(columns={'index': 'Step'}, inplace=True)

    # Plot using Plotly Express
    fig = px.line(df_long, x='Step', y='Cumulative Divergence', color='Article', markers=True,
                title="Stepwise Divergence from random path", color_discrete_map=color_dict)
    fig.add_hline(y=0, line=dict(color='black', dash='dash' ),  
                annotation_text="Random path", annotation_position="bottom right",  
                annotation_font=dict(size=12, color="black")  
    )

    fig.update_layout(
        xaxis=dict(
            range=[df_long['Step'].min() - 0.5, df_long['Step'].max() + 0.5],
            title='Step',
            tickmode='linear'  # Ensures all integer ticks are shown
        ),
        yaxis=dict(title='Value'),
        width=800,  # Set the width of the plot
        height=600  # Set the height of the plot
    )

    fig.show()

def plot_category_step_divergence(step_divergence, df_categories_filtered, color_dict):
    """
    Plots the stepwise deviation from a random path for different categories.
    Parameters:
    step_divergence (pd.DataFrame): DataFrame containing the stepwise divergence values for each article.
    df_categories_filtered (pd.DataFrame): DataFrame containing the article categories with at least 'article' and 'level_1' columns.
    color_dict (dict): Dictionary mapping category names to colors for the plot.
    Returns:
    None: Displays an interactive plotly line plot with markers.
    """
    

    #get level 1 categories for each articles
    cat_mean_div = step_divergence.transpose().reset_index()
    cat_mean_div = cat_mean_div.merge(right=df_categories_filtered[['article', 'level_1']], on='article', how='left')

    #sum probabilities of articles of the same category 
    cat_mean_div = cat_mean_div.groupby('level_1').agg('sum').reset_index().drop(columns='article')

    df_long_cat = cat_mean_div.melt(id_vars=['level_1'], var_name='Step', value_name='Value')

    # Convert the 'Step' column to start from 1
    df_long_cat['Step'] = df_long_cat['Step'].astype(int) + 1

    # Plot the data as a scatter plot with lines
    fig = px.line(
        df_long_cat,
        x='Step',
        y='Value',
        color='level_1',
        title='Stepwise deviation from random path, per category',
        color_discrete_map=color_dict,
        markers=True
    )

    # Add a horizontal line at y=0
    fig.add_hline(
        y=0,
        line=dict(color='black', dash='dash'),
        annotation_text="Random path",
        annotation_position="top right",
        annotation_font=dict(size=12, color="black")
    )

    # Adjust x-axis to add white space before and after the steps
    fig.update_layout(
        xaxis=dict(
            range=[df_long_cat['Step'].min() - 0.5, df_long_cat['Step'].max() + 0.5],
            title='Step',
            tickmode='linear'  # Ensures all integer ticks are shown
        ),
        yaxis=dict(title='Value'),
        width=800,  # Set the width of the plot
        height=600  # Set the height of the plot
    )

    fig.show()
