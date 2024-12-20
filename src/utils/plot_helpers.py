import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sn
import scipy.stats as stats
import numpy as np

def create_colored_treemap(labels, parents, values, ids, color_palette=None, title="Treemap", background_color='transparent'):
    """
    Creates a Plotly Treemap with colors propagated from level_1 to all children.

    Parameters:
    - labels (list): List of node labels.
    - parents (list): List of parent nodes.
    - values (list): List of values (used for proportional sizing).
    - ids (list): List of unique node IDs.
    - color_palette (dict): Dictionary mapping level_1 labels to colors. If None, a default palette is used.
    - title (str): Title of the treemap.
    - background_color (str): Background color of the plot ('white' or 'transparent').

    Returns:
    - fig (plotly.graph_objects.Figure): A Plotly Treemap figure.
    """

    # Function to propagate level_1 color to all children
    def get_colors_for_hierarchy(ids, color_palette):
        colors = []
        for tag in ids:
            # Extract the level_1 part of the label (before any slash '/')
            level_1 = tag.split('/')[0]
            # Get the color for level_1; default to light gray if not found
            color = color_palette.get(level_1, '#d3d3d3')
            colors.append(color)
        return colors

    # Generate colors for the hierarchy if palette is given
    colors = get_colors_for_hierarchy(ids, color_palette) if color_palette else None

    # Determine the background color settings
    if background_color == 'transparent':
        paper_bgcolor = 'rgba(0,0,0,0)'  # Transparent
        plot_bgcolor = 'rgba(0,0,0,0)'   # Transparent
    else:
        paper_bgcolor = background_color  # Solid color
        plot_bgcolor = background_color   # Solid color

    # Create the Treemap
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        ids=ids,
        marker=dict(colors=colors), # Apply colors if available
        textfont=dict(size=18),
        branchvalues='total'  # Ensures proportional sizing by summation of children
    ))

    # Update the layout with background color and title
    fig.update_layout(
        margin=dict(t=50, l=10, r=10, b=5),
        title=title,
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor
    )
    fig.show()

    return fig

def plot_position_line(S_T_fin_percentages_norm_steps, S_T_opt_fin_percentages_norm_steps, category_fin_means_norm, 
                       category_unf_means_norm, palette, title="Category Percentage Used at Each Step",background_color='white'):
    """
    Plot an interactive line plot of category frequencies across positions with both normalized and non-normalized views.
    Add bar plot as a subplot with inverted axes and hashed bars for 'unfinished'.
    
    Parameters:
        S_T_fin_percentages_norm_steps (DataFrame): DataFrame with user data.
        S_T_opt_fin_percentages_norm_steps (DataFrame): DataFrame with optimal data.
        category_fin_means_norm (DataFrame): Finished category percentages.
        category_unf_means_norm (DataFrame): Unfinished category percentages.
        palette (dict): Color palette for categories.
        title (str): Title of the plot.
        background_color (str): Background color of the plot ('white' or 'transparent').
    """
    # Extract and sort categories
    categories = S_T_fin_percentages_norm_steps["categories"].unique()
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3, 
        subplot_titles=("Users' paths", "Optimal paths", "Percentage difference across all steps"),
        horizontal_spacing=0.05
    )
    
    # Add non-normalized line plot traces (Users)
    for category in categories:
        category_data = S_T_fin_percentages_norm_steps[S_T_fin_percentages_norm_steps["categories"] == category]
        fig.add_trace(
            go.Scatter(
                x=category_data['step'], 
                y=category_data['percentage'],
                mode="lines+markers",
                name=category,
                line=dict(color=palette.get(category, 'grey'))
            ), row=1, col=1
        )
    
    # Add normalized line plot traces (Optimal)
    for category in categories:
        category_data = S_T_opt_fin_percentages_norm_steps[S_T_opt_fin_percentages_norm_steps["categories"] == category]
        fig.add_trace(
            go.Scatter(
                x=category_data['step'], 
                y=category_data['percentage'],
                mode="lines+markers",
                name=category,
                line=dict(color=palette.get(category, 'grey')),
                showlegend=False  # Show legend only on the first subplot
            ), row=1, col=2
        )
    
    # Combine datasets for bar plot
    category_fin_means_norm['path'] = 'finished'
    category_unf_means_norm['path'] = 'unfinished'
    concat_means_norm = pd.concat([category_fin_means_norm, category_unf_means_norm])

    # Prepare bar plot data
    for category in categories:
        bar_data_finished = concat_means_norm[(concat_means_norm['categories'] == category) & (concat_means_norm['path'] == 'finished')]
        bar_data_unfinished = concat_means_norm[(concat_means_norm['categories'] == category) & (concat_means_norm['path'] == 'unfinished')]
        
   # Prepare bar plot data
    bar_width = 0.4  # Set width for each bar
        
    for i, category in enumerate(categories):
        bar_data_finished = concat_means_norm[(concat_means_norm['categories'] == category) & (concat_means_norm['path'] == 'finished')]
        bar_data_unfinished = concat_means_norm[(concat_means_norm['categories'] == category) & (concat_means_norm['path'] == 'unfinished')]

        # Add finished bar
        fig.add_trace(
            go.Bar(
                x=[bar_data_finished['percentage_diff'].values[0]],
                y=[i],
                orientation='h',
                width=bar_width,
                marker=dict(color=palette.get(category, 'grey'), line=dict(width=0)),
                name='Finished', 
                showlegend=(i == 14)
            ), row=1, col=3
        )

        # Add unfinished bar with hashing
        fig.add_trace(
            go.Bar(
                x=[bar_data_unfinished['percentage_diff'].values[0]],
                y=[i + bar_width],
                orientation='h',
                width=bar_width,
                marker=dict(color=palette.get(category, 'grey'), pattern_shape="/"),
                name='Unfinished',  
                showlegend=(i == 14)
            ), row=1, col=3
        )

    # Determine the background color
    if background_color == 'transparent':
        paper_bgcolor = 'rgba(0,0,0,0)'  # Transparent
        plot_bgcolor = 'rgba(0,0,0,0)'   # Transparent
    else:
        paper_bgcolor = background_color  # Solid color
        plot_bgcolor = background_color   # Solid color

    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(title="Step",dtick=1),
        xaxis2=dict(title="Step",dtick=1),
        yaxis=dict(title="Percentage", automargin=True, range=[0, 50]),
        yaxis2=dict(automargin=True, range=[0, 50]),
        yaxis3=dict(autorange="reversed"),  # Reverse the y-axis order
        xaxis3_title="Percentage difference",
        template="plotly_white",
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor
    )

    # Remove y-axis for the third subplot
    fig.update_yaxes(showticklabels=False, title=None, row=1, col=3)

    # Show the plot
    fig.show()

    return fig

def plot_sankey_voyage(df, background_color='transparent'):
    """
    Plots a Sankey diagram to visualize the distribution of paths classified as 'Voyage' or 'Non-Voyage'.

    Parameters:
        df (DataFrame): DataFrame containing boolean 'Wiki_Voyage' and 'source_maincategory', 'target_maincategory' columns
        background_color (str): Background color of the plot ('white' or 'transparent').

    Returns:
        Figure: Sankey diagram as a Plotly Figure object.
    """

    # Mapping for start, voyage, and end nodes
    df_all_voyage = df.copy()
    df_all_voyage['source_category_label'] = df_all_voyage['source_maincategory'].apply(lambda x: 'Source is a World Regions' if x == 'World Regions' else 'Source is not a World Regions')
    df_all_voyage['target_category_label'] = df_all_voyage['target_maincategory'].apply(lambda x: 'Target is a World Regions' if x == 'World Regions' else 'Target is not a World Regions')
    df_all_voyage['voyage_label'] = df_all_voyage['Wikispeedia_Voyage'].apply(lambda x: 'Voyages' if x else 'Non-Voyages')

    # Start→Voyage flows
    start_voyage_flows = df_all_voyage.groupby(['source_category_label', 'voyage_label']).size().reset_index(name='count')

    # Voyage→End flows
    voyage_end_flows = df_all_voyage.groupby(['voyage_label', 'target_category_label']).size().reset_index(name='count')

    # Define node labels
    labels = ['Source is a World Regions', 'Source is not a World Regions',
              'Voyages', 'Non-Voyages',
              'Target is a World Regions', 'Target is not a World Regions']

    # Create mappings for source and target node indices
    label_map = {label: i for i, label in enumerate(labels)}

    # Initialize lists for diagram data
    sources = []
    targets = []
    values = []

    # Add Start→Voyage flows
    for _, row in start_voyage_flows.iterrows():
        sources.append(label_map[row['source_category_label']])
        targets.append(label_map[row['voyage_label']])
        values.append(row['count'])

    # Add Voyage→End flows
    for _, row in voyage_end_flows.iterrows():
        sources.append(label_map[row['voyage_label']])
        targets.append(label_map[row['target_category_label']])
        values.append(row['count'])

    # Define node colors
    node_colors = [
        '#2CB5AE',  # First in Countries/Geography
        '#4b4b4b',  # First not in Countries/Geography
        '#2CB5AE',  # Voyage
        '#4b4b4b',  # Non-Voyage
        '#2CB5AE',  # Target in Countries/Geography
        '#4b4b4b'   # Target not in Countries/Geography
    ]

    link_colors = [
        'rgba(75, 75, 75, 0.3)',  # not voyage
        'rgba(75, 75, 75, 0.3)',  # not voyage
        'rgba(44, 181, 174, 0.3)',  # Voyage
        'rgba(75, 75, 75, 0.3)',  # not voyage
        'rgba(75, 75, 75, 0.3)',  # not voyage
        'rgba(44, 181, 174, 0.3)'  # Voyage
    ]

    # Determine the background color
    if background_color == 'transparent':
        paper_bgcolor = 'rgba(0,0,0,0)'  # Transparent
        plot_bgcolor = 'rgba(0,0,0,0)'   # Transparent
    else:
        paper_bgcolor = background_color  # Solid color
        plot_bgcolor = background_color   # Solid color

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=20, thickness=20, line=dict(color="white", width=0), label=labels, color=node_colors),
        link=dict(source=sources, target=targets, value=values, color=link_colors) #color='rgba(60,60,60,0.3)
    )])
    
    fig.update_layout(
        title_text="Voyage and Non-Voyage Paths",
        font_size=10,
        title_font_size=14,
        title_x=0.5,
        paper_bgcolor=paper_bgcolor,  #remove to set bg white 
        plot_bgcolor=plot_bgcolor   #remove to set bg white 
    )
    
    return fig

def plot_articles_pie_chart(df, palette, abbreviations=None):
    """
    Plots a simplified pie chart of the total number of articles per Level 1 category.

    Parameters:
    - df (DataFrame): The DataFrame containing 'article' and 'level_1' columns.
    - abbreviations (dict, optional): A dictionary mapping full category names to abbreviations.
    """
    # Group by Level 1 category and count the number of articles
    category_counts = df['level_1'].value_counts()
    category_counts = category_counts.sort_values(ascending=True)

    # Handle small categories (less than 3%) by grouping them as 'Others'
    threshold = 3  # percentage threshold
    small_categories = category_counts[category_counts / category_counts.sum() * 100 < threshold]
    small_categories_total = small_categories.sum()
    large_categories = category_counts[category_counts / category_counts.sum() * 100 >= threshold]

    # Add "Others" for small categories
    if not small_categories.empty:
        others = pd.Series({f'Others': small_categories_total})
        large_categories = pd.concat([large_categories, others])

    # Prepare the labels: Use abbreviations if provided
    if abbreviations:
        labels = [abbreviations.get(cat, cat) for cat in large_categories.index]
        legend_labels = [f"{cat} ({abbreviations.get(cat, 'N/A')})" for cat in large_categories.index]
    else:
        labels = large_categories.index
        legend_labels = labels

    # Plot the pie chart
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        large_categories, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=90,
        pctdistance=0.8,
        colors = [palette.get(label, '#cccccc') for label in large_categories.index]
    )

    # Customize the font and color of the numbers
    for autotext in autotexts:
        autotext.set_fontsize(9)  # Change font size

    # Set the title of the plot
    ax.set_title('Articles Distribution per Level 1 Category')

    # Place the legend outside the pie chart to avoid overlap
    ax.legend(
        legend_labels, 
        title="Categories", 
        loc='center left', 
        bbox_to_anchor=(1, 0.5), 
        fontsize=10
    )

    # Display the pie chart
    plt.tight_layout()  # Adjust layout to ensure everything fits
    plt.show()
    
def plot_proportion_links_in_cat_pie_chart(df, in_or_out , palette, abbreviations=None):
    """
    use plot_proportions_of_in_and_out_degree_in_categories() for both on same plot 
    Plots pie chart of the total number (sum) of all links for category.
    ex for out degree: 25% of all links are in country articles 
    ex for in degree: 25% of all links target country articles 
    Parameters:
    - df (DataFrame): The DataFrame containing 'article' and 'category', 'in_degree', and 'out_degree' columns.
    - in_else_out : in degree if true, out degree if false
    - pallette (dict): palette to use for categories containing others country and geo
    - abbreviations (dict, optional): A dictionary mapping full category names to abbreviations.

    """
    if in_or_out:
        in_degree_tot = df.groupby('category')['in_degree'].sum().sort_values(ascending=False)
    else :
        in_degree_tot = df.groupby('category')['out_degree'].sum().sort_values(ascending=False)
    
    labels_cat = in_degree_tot.keys()

    # Handle small categories (less than 3%) by grouping them as 'Others'
    threshold = 3  # percentage threshold
    small_categories = in_degree_tot[in_degree_tot / in_degree_tot.sum() * 100 < threshold]
    small_categories_total = small_categories.sum()
    large_categories = in_degree_tot[in_degree_tot / in_degree_tot.sum() * 100 >= threshold]

    # Add "Others" for small categories
    if not small_categories.empty:
        others = pd.Series({f'Others': small_categories_total})
        large_categories = pd.concat([large_categories, others])

    # Prepare the labels: Use abbreviations if provided
    if abbreviations:
        labels = [abbreviations.get(cat, cat) for cat in large_categories.index]
        legend_labels = [f"{cat} ({abbreviations.get(cat, 'N/A')})" for cat in large_categories.index]
    else:
        labels = large_categories.index
        legend_labels = labels 

    # Plot the pie chart
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        large_categories, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=90,
        pctdistance=0.8,
        colors=[palette[label] for label in labels_cat]
    )

    # Customize the font and color of the numbers
    for autotext in autotexts:
        autotext.set_fontsize(9)  # Change font size

    # Set the title of the plot
    if in_or_out:
        ax.set_title('Category-wise share of all out-degree links')
    else :
        ax.set_title('Proportions of links targetting each categories')
        
    # Place the legend outside the pie chart to avoid overlap
    ax.legend(
        legend_labels, 
        title="Categories", 
        loc='center left', 
        bbox_to_anchor=(1, 0.5), 
        fontsize=10
    )

    # Display the pie chart
    plt.tight_layout()  # Adjust layout to ensure everything fits
    plt.show()
    
def plot_proportions_of_in_and_out_degree_in_categories(df, palette, abbreviations=None, threshold=3):
    """
    Plots pie charts for proportions of in-degree and out-degree links across categories.

    Parameters:
    - df (DataFrame): The DataFrame containing 'category', 'in_degree', and 'out_degree' columns.
    - palette (dict): Palette to use for categories, including 'Others'.
    - abbreviations (dict, optional): A dictionary mapping full category names to abbreviations.
    - threshold (int, optional): Minimum percentage to display a category individually; others are grouped under 'Others'.
    """
    # Sum in-degree and out-degree by category
    in_degree_tot = df.groupby('category')['in_degree'].sum()
    out_degree_tot = df.groupby('category')['out_degree'].sum()

    # Handle small categories by grouping them as "Others"
    def handle_small_categories(category_dict):
        small_categories = category_dict[category_dict / category_dict.sum() * 100 < threshold]
        small_categories_total = small_categories.sum()
        large_categories = category_dict[category_dict / category_dict.sum() * 100 >= threshold]
        if not small_categories.empty:
            others = pd.Series({'Others': small_categories_total})
            large_categories = pd.concat([large_categories, others])
        return large_categories

    in_degree_tot = handle_small_categories(in_degree_tot)
    out_degree_tot = handle_small_categories(out_degree_tot)

    # Prepare labels
    def prepare_labels(category_dict):
        if abbreviations:
            labels = [abbreviations.get(cat, cat) for cat in category_dict.index]
        else:
            labels = category_dict.index
        return labels

    in_labels = prepare_labels(in_degree_tot)
    out_labels = prepare_labels(out_degree_tot)

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Plot in-degree pie chart
    ax[0].pie(
        in_degree_tot.values,
        labels=in_labels,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.8,
        colors=[palette.get(cat, "#cccccc") for cat in in_degree_tot.index],
    )
    ax[0].set_title("Proportion of Links Targeting Each Category", fontsize=14)

    # Plot out-degree pie chart
    ax[1].pie(
        out_degree_tot.values,
        labels=out_labels,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.8,
        colors=[palette.get(cat, "#cccccc") for cat in out_degree_tot.index],
    )
    ax[1].set_title("Proportion of Links Leaving Each Category", fontsize=14)

    # Add a single legend
    used_categories = df["category"].unique()
    used_palette = {key: palette[key] for key in used_categories}
    add_legend_category(
        fig=fig,
        palette_category=used_palette,
        bbox_to_anchor=(1, 0.75)
    )

    # Adjust layout
    plt.suptitle("Proportions of in and out degree links by Category", fontsize=16, y=1.05)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust for the legend
    plt.show()
    
def add_legend_category(fig, palette_category, bbox_to_anchor=(1.15, 0.85)):
    """
    Adds a unique legend to the figure for the categories.
    
    Parameters:
        fig (Figure): The figure to add the legend to.
        palette_category (dict): Color palette for the categories.
        bbox_to_anchor (tuple, optional): The position of the legend.
    """

    handles = [
        plt.Line2D([0], [0], marker='o', color=color, linestyle='', markersize=10) 
        for color in palette_category.values()  # Use values from the dictionary
    ]
    labels = list(palette_category.keys())
    fig.legend(
        handles, 
        labels, 
        bbox_to_anchor=bbox_to_anchor, 
        title="Categories", 
    )
    
def plot_proportion_category_start_stop_pies(df_article, palette, abbreviations=None, threshold=2.3):
    """
    Makes pie charts showing the proportion of categories in start/target articles.
    
    Parameters:
    - df_article (DataFrame): DataFrame containing 'start_count' and 'target_count'
    - palette (dict): Color palette for the categories
    - abbreviations (dict, optional): Dictionary mapping full category names to abbreviations 
    - threshold (int, optional): Minimum percentage to display a category individually: Others are grouped under 'Others'
    """
    # count number of articles given as start and target for each category
    start_dict = df_article.groupby('category')['start_count'].sum()
    target_dict = df_article.groupby('category')['target_count'].sum()

    # Handle small categories by grouping them as "Others"
    def handle_small_categories(category_dict):
        small_categories = category_dict[category_dict / category_dict.sum() * 100 < threshold]
        small_categories_total = small_categories.sum()
        large_categories = category_dict[category_dict / category_dict.sum() * 100 >= threshold]
        if not small_categories.empty:
            others = pd.Series({'Others': small_categories_total})
            large_categories = pd.concat([large_categories, others])
        return large_categories

    start_dict = handle_small_categories(start_dict)
    target_dict = handle_small_categories(target_dict)

    # Use abbreviations if provided
    def prepare_labels(category_dict):
        """
        Prepare labels for the pie chart.
        
        Parameters:
            category_dict (Series): Series containing category counts.
        
        """
        if abbreviations:
            labels = [abbreviations.get(cat, cat) for cat in category_dict.index]
            legend_labels = [f"{cat} ({abbreviations.get(cat, 'N/A')})" for cat in category_dict.index]
        else:
            labels = category_dict.index
            legend_labels = labels
        return labels, legend_labels

    start_labels, _ = prepare_labels(start_dict)
    target_labels, _ = prepare_labels(target_dict)

    # subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Plot start atricles pie in 0 slot
    ax[0].pie(
        start_dict.values,
        labels=start_labels,
        colors=[palette.get(cat, "#cccccc") for cat in start_dict.index],
        autopct='%1.1f%%',
        #textprops={'color':"w"}
        startangle=90,
        pctdistance=0.8
    )
    ax[0].set_title("Proportion of categories in source articles", fontsize=14)

    # Plot target articles pie in 1 slot
    ax[1].pie(
        target_dict.values,
        labels=target_labels,
        colors=[palette.get(cat, "#cccccc") for cat in target_dict.index],
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.8
    )
    ax[1].set_title("Proportion of categories in target articles", fontsize=14)

    used_categories = df_article["category"].unique()
    used_palette = {key: palette[key] for key in used_categories}
    add_legend_category(
        fig=fig,
        palette_category=used_palette,
        bbox_to_anchor=(1.15, 0.7)
    )
    plt.suptitle("Proportions of Categories in Source and Target Articles", fontsize=16, y=1.05)
    plt.tight_layout()

    plt.show()

def plot_metrics_by_category(df_article, metrics, palette_category_dict, category_abbreviations):
    """
    Plots bar charts for multiple metrics by category using Plotly.

    Parameters:
    - df_article (DataFrame): DataFrame containing article data.
    - metrics (list): List of metric column names to plot.
    - palette_category_dict (dict): Color palette for the categories.
    - category_abbreviations (dict): Abbreviations for categories.
    """
    # Loop through metrics and plot
    fig, ax = plt.subplots(2, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        row, col = divmod(i, 3)
        order = df_article.groupby("category")[metric].mean().sort_values(ascending=False).reset_index()["category"]
        sn.barplot(
            x="category", 
            y=metric, 
            hue="category", 
            palette=palette_category_dict, 
            data=df_article, 
            ax=ax[row, col], 
            order=order
        )
        ax[row, col].set_title(f'{metric.replace("_", " ").capitalize()} by Category')
        ax[row, col].set_xticklabels([])
        if row == 0 :
            ax[row, col].set_xlabel('')

    used_categories = df_article["category"].unique()
    used_palette = {key: palette_category_dict[key] for key in used_categories}
    add_legend_category(fig, used_palette)

    plt.suptitle("Articles Complexity by Categories", y=1, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_article_popularity_link_density(df_article, df_finished_voyage, palette_category_dict, df_categories_filtered):
    """
    Plots bar charts for article popularity and link density by category.
    
    Parameters:
        df_article (DataFrame): DataFrame containing article data.
        df_finished_voyage (DataFrame): DataFrame containing finished voyage data.
        palette_category_dict (dict): Color palette for the categories.
        df_categories_filtered (DataFrame): DataFrame containing category data for each article.
       
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    #Plot the most visited articles in finished paths
    all_articles = []
    df_finished_voyage['path'].apply(lambda x: all_articles.extend(x.split(';')))
    df_path_articles = pd.Series(all_articles).value_counts().rename_axis('article_name').reset_index(name='value_counts')
    df_path_articles["category"]=df_path_articles["article_name"].apply(lambda x: df_categories_filtered[df_categories_filtered["article"]==x]["level_1"].values[0] if len(df_categories_filtered[df_categories_filtered["article"]==x]["category"].values)>0 else "None")
    df_path_articles = df_path_articles[df_path_articles['article_name'] != '<']

    sn.barplot(x='value_counts', y='article_name', hue="category", palette=palette_category_dict, data=df_path_articles.head(15), ax=ax[0])
    ax[0].set_title('Most visited articles in paths')
    ax[0].legend_.remove() 

    for i, metric in enumerate(["in_degree", "out_degree"]):
        sn.barplot(x=metric, y='article', hue="category", palette=palette_category_dict, data=df_article.sort_values(metric, ascending=False).head(15), ax=ax[i+1])
        ax[i+1].set_title(f'Articles with the most links ({metric.replace("_", " ").capitalize()}) (without duplicates)')
        ax[i+1].legend_.remove()
        ax[i+1].set_ylabel('')

    used_categories = df_article["category"].unique()
    used_palette = {key: palette_category_dict[key] for key in used_categories}
    add_legend_category(fig, used_palette)
    plt.suptitle("Relationshio between article popularity and link density", y=1, fontsize=16)
    plt.tight_layout()
    plt.show()

def remove_outliers(df, col):
    """
    Removes outliers from a DataFrame based on a specified column.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    filtered_df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
    return filtered_df

def plot_difficulties_voyage (df_finished, df_unfinished, palette_category_dict):
    """
    Plots a comparison of difficulty metrics between Voyage and Non-Voyage games.
    
    Parameters:
    - df_finished (DataFrame): DataFrame containing finished game data.
    - df_unfinished (DataFrame): DataFrame containing unfinished game data.
    - palette_category_dict (dict): Color palette for the categories.

    Returns:
    - Figure: Plotly figure object.
    """
    color_voyage = palette_category_dict['Voyages']
    
    df_finished_voyage = df_finished.copy()
    df_unfinished_voyage = df_unfinished.copy()

    df_finished_voyage["finished"] = True
    df_finished_voyage["cte"] = 1
    df_unfinished_voyage["finished"] = False
    df_unfinished_voyage["cte"] = 1
    df_voyage = pd.concat([df_finished_voyage, df_unfinished_voyage])

    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=(
            "Duration Distribution", 
            "Completion Ratios", 
            "Rating Distribution for Voyage Game", 
            "Rating Distribution for Non-Voyage Game"
        )
    )

    # ==== PLOT 1 (Violin Plot: Duration Distribution) ====
    df_voyage_duration = df_finished_voyage[df_finished_voyage["Wikispeedia_Voyage"] == True]
    df_voyage_duration = remove_outliers(df_voyage_duration, "durationInSec")

    df_non_voyage_duration = df_finished_voyage[df_finished_voyage["Wikispeedia_Voyage"] == False]
    df_non_voyage_duration = remove_outliers(df_non_voyage_duration, "durationInSec")

    fig.add_trace(
        go.Violin(
            x=df_voyage_duration["cte"], 
            y=df_voyage_duration["durationInSec"],
            legendgroup="Yes", 
            scalegroup="Yes", 
            name="Voyage",
            side="negative", 
            line_color=color_voyage, 
            box_visible=True,
            meanline_visible=True,
            showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Violin(
            x=df_non_voyage_duration["cte"],
            y=df_non_voyage_duration["durationInSec"],
            legendgroup="No", 
            scalegroup="No", 
            name="Non-Voyage",
            side="positive", 
            line_color="gray",
            box_visible=True,
            meanline_visible=True,
            showlegend=False
        ),
        row=1, col=1
    )
    _, p_value = stats.ttest_ind(df_voyage_duration["durationInSec"].dropna(), df_non_voyage_duration["durationInSec"].dropna())
    fig.add_annotation(x=1, y= df_voyage_duration["durationInSec"].max() + 50, text=convert_pvalue_to_asterisks(p_value), showarrow=False, font=dict(color='black'))
    fig = custom_legend_pval(fig, title = False, y_pos = 0.8, x_pos = 1.03, id = 0.02)

    # Axis labels
    fig.update_yaxes(title_text="Duration (seconds)", row=1, col=1)

    # ==== PLOT 2 (Bar Plot: Completion Ratios) ====
    df_voyage_comparison = df_voyage.groupby(["finished", "Wikispeedia_Voyage"])[["Wikispeedia_Voyage"]].count() 
    df_voyage_comparison.columns = ["count"]
    df_voyage_comparison = df_voyage_comparison.reset_index()
    df_voyage_comparison = df_voyage_comparison.sort_values(by="finished", ascending=False)
    df_voyage_comparison["percentage"] = df_voyage_comparison.groupby("Wikispeedia_Voyage")["count"].transform(lambda x: (x / x.sum()) * 100).round(1)
    df_voyage_comparison["voyage_label"] = df_voyage_comparison["Wikispeedia_Voyage"].map({False: "Non-Voyage", True: "Voyage"})
    df_voyage_comparison["finished_label"] = df_voyage_comparison["finished"].map({False: "Unfinished", True: "Finished"})

    for voyage_label, color in [("Voyage", color_voyage), ("Non-Voyage", 'gray')]:
        filtered_data = df_voyage_comparison[df_voyage_comparison["voyage_label"] == voyage_label]
        fig.add_trace(
            go.Bar(
                x=filtered_data["finished_label"],
                y=filtered_data["percentage"],
                text=filtered_data["count"],
                name=voyage_label,
                marker_color=color,
                texttemplate="Count: %{text}",
            ),
            row=1, col=2
        )
        
    # Axis labels
    fig.update_yaxes(title_text="Pourcentage (%)", row=1, col=2)
    fig.update_xaxes(title_text="Path Type", row=1, col=2)    
        
    # ==== PLOT 3 (Bar Plot: Rating Distribution for Voyage Games) ====
    df_voyage_rating = df_finished_voyage[df_finished_voyage["Wikispeedia_Voyage"] == True].copy()
    df_voyage_rating["rating"] = df_voyage_rating["rating"].fillna('NaN').astype(str)
    df_voyage_rating["have_back_click"] = df_voyage_rating["back_clicks"] > 0
    back_click_per_rating = pd.DataFrame(df_voyage_rating.groupby("rating")["have_back_click"].mean()).reset_index() # count the number of path with a rating for each rating
    df_voyage_rating = df_voyage_rating.groupby("rating")["rating"].count().reset_index(name="count")
    df_voyage_rating["pourcent"] = df_voyage_rating["count"] / df_voyage_rating["count"].sum() * 100
    df_voyage_rating["back_clicks"] = back_click_per_rating["have_back_click"].values
    df_voyage_rating["Back-Click"] = df_voyage_rating["back_clicks"]*df_voyage_rating["pourcent"]
    df_voyage_rating["Without Back-Click"] = (1 - df_voyage_rating["back_clicks"])*df_voyage_rating["pourcent"]
    df_voyage_rating = pd.melt(df_voyage_rating, id_vars=['rating', 'count', 'pourcent'], value_vars=['Back-Click', 'Without Back-Click'], var_name='back_click')

    df = df_voyage_rating[df_voyage_rating["back_click"]== 'Back-Click']
    fig.add_trace(go.Bar(x=df["rating"],
                            y=df["value"],
                            marker_color=color_voyage,
                            marker_pattern_shape="/",
                            offsetgroup=0,
                            legendgroup='Back-Click',
                            name = 'Back-Click'),
                    row=2, col=1)
    df = df_voyage_rating[df_voyage_rating["back_click"]== 'Without Back-Click']
    fig.add_trace(go.Bar(x=df["rating"],
                            y=df["value"],
                            marker_color=color_voyage,
                            offsetgroup=0,
                            base = df_voyage_rating[df_voyage_rating["back_click"]== 'Back-Click']["value"],
                            legendgroup='Without Back-Click',
                            name = 'Without Back-Click'),
                    row=2, col=1)

    # axis labels
    fig.update_yaxes(title_text="Pourcentage", row=2, col=1)
    fig.update_xaxes(title_text="Rating", row=2, col=1)

    # # ==== PLOT 4 (Bar Plot: Rating Distribution for Non-Voyage Games) ====
    df_non_voyage_rating = df_finished_voyage[df_finished_voyage["Wikispeedia_Voyage"] == False].copy()
    df_non_voyage_rating["rating"] = df_non_voyage_rating["rating"].fillna('NaN').astype(str)
    df_non_voyage_rating["have_back_click"] = df_non_voyage_rating["back_clicks"] > 0
    back_click_per_rating = pd.DataFrame(df_non_voyage_rating.groupby("rating")["have_back_click"].mean()).reset_index() # count the number of path with a rating for each rating
    df_non_voyage_rating = df_non_voyage_rating.groupby("rating")["rating"].count().reset_index(name="count")
    df_non_voyage_rating["pourcent"] = df_non_voyage_rating["count"] / df_non_voyage_rating["count"].sum() * 100
    df_non_voyage_rating["back_clicks"] = back_click_per_rating["have_back_click"].values
    df_non_voyage_rating["Back-Click"] = df_non_voyage_rating["back_clicks"]*df_non_voyage_rating["pourcent"]
    df_non_voyage_rating["Without Back-Click"] = (1 - df_non_voyage_rating["back_clicks"])*df_non_voyage_rating["pourcent"]
    df_non_voyage_rating = pd.melt(df_non_voyage_rating, id_vars=['rating', 'count', 'pourcent'], value_vars=['Back-Click', 'Without Back-Click'], var_name='back_click')

    df = df_non_voyage_rating[df_non_voyage_rating["back_click"]== 'Back-Click']
    fig.add_trace(go.Bar(x=df["rating"],
                            y=df["value"],
                            marker_color="gray",
                            marker_pattern_shape="/",
                            offsetgroup=0,
                            showlegend=False,
                            name = 'Back-Click'),
                    row=2, col=2)
    df = df_non_voyage_rating[df_non_voyage_rating["back_click"]== 'Without Back-Click']
    fig.add_trace(go.Bar(x=df["rating"],
                            y=df["value"],
                            marker_color="gray",
                            offsetgroup=0,
                            base = df_non_voyage_rating[df_non_voyage_rating["back_click"]== 'Back-Click']["value"],
                            showlegend=False,
                            name = 'Without Back-Click'),
                    row=2, col=2)

    # axis labels
    fig.update_yaxes(title_text="Pourcentage", row=2, col=2)
    fig.update_xaxes(title_text="Rating", row=2, col=2)

    # ==== Final Layout Update ====
    fig.update_layout(
        height=1000, width=1000,  # Adjust size of the overall figure
        title="Summary of Voyage and Non-Voyage Game Metrics",
        showlegend=True,
        legend_title="",
        xaxis_title="Game Type",
        yaxis_title="Count/Percentage",
        violingap=0.4, 
        violinmode="overlay",
    )
    fig.show()

def convert_pvalue_to_asterisks(pvalue):
    """
    Converts a p-value to asterisks for significance levels
    
    Parameters:
    - pvalue (float): The p-value to convert
    
    Returns:
    - str: The converted p-value with asterisks
    """
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    if pvalue <= 0.05:
        return "*"
    return "ns"

def perform_ttest(df, model_name1, model_name2):
    """Performs t-tests and returns p-values for each answer between models"""
    _, p_value = stats.ttest_ind(df[(df['category'] == model_name1)]["position"].dropna(), df[(df['category'] == model_name2)]["position"].dropna())
    return p_value 

def annotate_pvalues_combinaison(fig, p_values, x_labels, bar_positions, bar_heights, annotation = "line", h = 0.02, combinaison_nb=None):
    """
    Annotate p-values on the plot.
    
    Parameters:
    - fig (Figure): The plotly figure to annotate.
    - p_values (dict): Dictionary of p-values to annotate.
    - x_labels (list): List of x-axis labels.
    - bar_positions (list): List of bar positions.
    - bar_heights (list): List of bar heights.
    - annotation (str): Annotation type ('line' or 'top').
    - h (float): Height of the annotation line.
    - combinaison_nb (int): Number of combinaisons.
    
    Returns:
    - Figure: The annotated plotly figure.
    """

    for i, ((x_label1, x_label2), p_value) in enumerate(p_values.items()):
        
        index1 = np.where(x_labels == x_label1)[0][0]
        index2 = np.where(x_labels == x_label2)[0][0]

        x1 = bar_positions[index1]
        x2 = bar_positions[index2]

        max_height = max(bar_heights)
        
        if annotation == "line" :
            y = max_height + 0.1 + (i - (index1* combinaison_nb))/15
            fig.add_shape(type='line', x0=x1, y0=y, x1=x1, y1=y + h, line=dict(color='black'), xref='x', yref='y')
            fig.add_shape(type='line', x0=x1, y0=y + h, x1=x2, y1=y + h, line=dict(color='black'), xref='x', yref='y')
            fig.add_shape(type='line', x0=x2, y0=y + h, x1=x2, y1=y, line=dict(color='black'), xref='x', yref='y')
            
            fig.add_annotation(x=(x1 + x2) * .5, y= y + h+ 0.03, text=convert_pvalue_to_asterisks(p_value), showarrow=False, font=dict(color='black'))
        elif annotation == "top" :
            fig.add_annotation(x=x2+0.1, y= max_height + 0.07, text=convert_pvalue_to_asterisks(p_value), showarrow=False, font=dict(color='black'))

def custom_legend_pval(fig, title = True, y_pos = 0.5, x_pos = 1.02, id = 0.05):
    """
    Add a custom legend for p-values.
    
    Parameters:
    - fig (Figure): The plotly figure to annotate.
    - title (bool): Whether to include a title.
    - y_pos (float): Y-position of the legend.
    - x_pos (float): X-position of the legend.
    - id (float): Height of each p-value label.
    
    Returns:
    - Figure: The annotated plotly figure.
    """
    p_value_labels = [
        "****: p-value < 0.0001",
        "***: p-value < 0.001",
        "**: p-value < 0.01",
        "*: p-value < 0.05",
        "ns: p-value >= 0.05"
    ]
    
    if title : 
        fig.add_annotation(
                x=x_pos,  # Position outside the plot area
                y= y_pos + 0.1,
                text="T-test P-values:",
                showarrow=False,
                xanchor="left",
                xref="paper",
                yref="paper",
                align="left",
                font=dict(size=13, color="black"),
            )
    for i, label in enumerate(p_value_labels):
        fig.add_annotation(
            x=x_pos,  # Position outside the plot area
            y= y_pos - i * id,  # Adjust spacing between labels
            text=label,
            showarrow=False,
            xanchor="left",
            xref="paper",
            yref="paper",
            align="left",
            font=dict(size=12, color="black"),
        )
    return fig
    
def plot_comparison_category_click_position(df_merged, df_category_position, colors = {'Clicked Link Position': '#AFD2E9', "Link Position to a category": '#9A7197'}):
    """
    Plots a comparison of the position of the clicked link in articles compared to the position of each category in articles.
    
    Parameters:
    - df_merged (DataFrame): DataFrame containing merged data.
    - df_category_position (DataFrame): DataFrame containing category position data.
    - colors (dict): Color palette for the plot.

    Returns:
    - Figure: Plotly figure object.
    """
    df_merged["category"] = "All"
    df_merged["Legend :"] = "Clicked Link Position"
    df_melted = pd.melt(df_category_position, var_name='category', value_name='position').dropna()
    df_melted["Legend :"] = "Link Position to a category"
    df_comparison_path_category = pd.concat([df_merged[["category", "position", "Legend :"]], df_melted])

    fig = px.box(df_comparison_path_category, x="category", y="position", color="Legend :", title="Position of the clicked link in articles compared to position of each category in articles", color_discrete_map=colors)
    fig.update_xaxes(tickangle=45)
    
    categories_name = df_comparison_path_category['category'].unique()

    p_values = {}
    for i in range(len(categories_name)):
        if categories_name[i] != "All":
            p_values[("All", categories_name[i])] = perform_ttest(df_comparison_path_category, "All", categories_name[i])   
    
    bar_pos = np.arange(len(categories_name)) + 0.17 # add 0.17 = the space between the 2 color categories
    bar_pos[0] -= 0.34
    bar_h= np.ones(len(categories_name))
    annotate_pvalues_combinaison(fig, p_values, categories_name,  bar_positions = bar_pos, bar_heights = bar_h, combinaison_nb = len(p_values.keys()), annotation = "top")
    fig = custom_legend_pval(fig, title = False, y_pos = 0.65, x_pos=1.03)

    fig.update_layout(
        autosize=False,
        width=1500,
        height=500,
        boxgroupgap=0.2, # update
        boxgap=0,
        xaxis=dict(
        title="Category",
        # tickvals=[-0.25, 1.25, 2.25]
        ),
        yaxis=dict(
        title="Position",
        ),)
    
    fig.show()

def plot_donut_link_position(df_merged, colors):
    """
    Plots a comparison of the mean link positions in articles and the mean click positions in all paths.
    
    Parameters:
    - df_merged (DataFrame): DataFrame containing merged data.
    - colors (dict): Color palette for the plot.
    
    Returns:
    - Figure: Plotly figure object.
    """

    df_info = df_merged[["total_links", "total_link_in_abstract", "total_link_in_infobox", "link_in_core", "link_in_abstract", "link_in_infobox"]].copy()
    df_info["total_link_core"] = df_info["total_links"] - df_info["total_link_in_abstract"] - df_info["total_link_in_infobox"]
    df_info["link_all"] = df_info["link_in_core"] + df_info["link_in_abstract"] + df_info["link_in_infobox"]
    df_info

    df_ratio_path = df_info[["total_link_in_abstract", "total_link_in_infobox", "total_link_core"]].div(df_info["total_links"], axis=0)
    df_ratio_path = df_ratio_path.mean()
    df_ratio_click = df_info[["link_in_abstract", "link_in_infobox", "link_in_core"]].div(df_info["link_all"], axis=0)
    df_ratio_click = df_ratio_click.mean()

    data = [

        go.Pie(values=[df_ratio_path["total_link_core"], df_ratio_path["total_link_in_abstract"], df_ratio_path["total_link_in_infobox"]],
        labels=['Body','Abstract', "Info-box"],
        domain={'x':[0.3,0.7], 'y':[0.16,0.8]}, 
        hole=0.5,
        direction='clockwise',
        sort=False,
        title=dict(
                text="Link position<br>in articles",
                font=dict(size=16)
            ),
        texttemplate="%{percent:.0%}",
        marker={'colors':colors},), 

        go.Pie(values=[df_ratio_click["link_in_core"] ,df_ratio_click["link_in_abstract"] ,df_ratio_click["link_in_infobox"]],
            labels=['Body','Abstract', "Info-box"],
            domain={'x':[0.1,0.9], 'y':[0,1]},
            hole=0.75,
            direction='clockwise',
            sort=False,
            marker={'colors':colors},
            title=dict(
                text="Link position in clicks",
                font=dict(size=16)
            ),
            titleposition='top center',
            texttemplate="%{percent:.0%}",
            showlegend=True)
    ]
        
    figure=go.Figure(data=data, layout={'title':'Comparison of Mean Link Positions in Articles <br>and the Clicks Positions in All Paths', 'width': 800, 'height': 600})

    figure.show()

def plot_duration_by_rating(df_finished, df_finished_strNaN, metric):
    """
    Plots the path duration by rating of the finished paths.
    
    Parameters:
    - df_finished (DataFrame): DataFrame containing finished game data.
    - df_finished_strNaN (DataFrame): DataFrame containing finished game data with NaN rating.
    - metric (str): The metric to plot.
    """
    plt.figure(figsize=(10, 6))
    blues_palette = sn.color_palette("Blues", n_colors=6)

    ax1 = plt.subplot(231)
    sn.histplot(df_finished[df_finished_strNaN['rating']=='NaN'], x=metric, bins=50, log_scale=True, color=blues_palette[0]) 
    mean = df_finished[df_finished_strNaN['rating']=='NaN'][metric].mean()
    plt.axvline(mean, color='red', label=f'Mean: {mean:.2f}', linestyle='--')
    plt.yscale('log')
    plt.xlabel('Duration in seconds')
    plt.title('NaN')
    plt.legend()

    for i in range(1, 6):
        plt.subplot(231+i, sharex = ax1, sharey=ax1)
        sn.histplot(df_finished[df_finished_strNaN['rating']==i], x=metric, bins=50, log_scale=True, color=blues_palette[i])
        mean = df_finished[df_finished_strNaN['rating']==i][metric].mean()
        plt.axvline(mean, color="red", label=f'Mean: {mean:.2f}', linestyle='--')
        plt.yscale('log')
        plt.title(i)
        plt.xlabel('Duration in seconds')
        plt.legend()

    plt.tight_layout()
    plt.suptitle('Path duration by rating of the finished paths', y=1.04)
    plt.show()
    
def extract_back_click_categories(category_path):
    """
    Extract categories where back-clicks occurred using Transition Category Path.
    
    Parameters:
    - category_path (list): List of categories in the path.
    
    Returns:
    - list: List of categories where back-clicks occurred.
    """
    back_click_categories = []
    for i, category in enumerate(category_path):
        if category == '<' and i > 0:  # Detect back-click and ensure valid index
            back_click_categories.append(category_path[i - 1])  # Use preceding category
    return back_click_categories

def plot_back_clicks(df_finished,rating_order_plot2, rating_colors, rating_order_plot1):
    """
    Plots the distribution of back-clicks per rating and the normalized back-clicks by category and rating.
    
    Parameters:
    - df_finished (DataFrame): DataFrame containing finished game data.
    - rating_order_plot2 (list): Order of ratings for the second plot.
    - rating_colors (dict): Color palette for the ratings.
    - rating_order_plot1 (list): Order of ratings for the first plot.
    """
    
    # Apply the function to identify back-click categories
    df_finished['back_click_categories'] = df_finished['Transition Category Path'].apply(extract_back_click_categories)

    # Explode back-click categories for analysis
    df_back_clicks = df_finished.explode('back_click_categories').dropna(subset=['back_click_categories'])
    df_back_clicks['rating'] = df_back_clicks['rating'].fillna('NaN')
    df_back_clicks['rating'] = df_back_clicks['rating'].astype(str)

    # Count total back-click occurrences grouped by category and rating
    category_rating_back_click_counts = (df_back_clicks.groupby(['back_click_categories', 'rating']).size().reset_index(name='back_click_count'))

    # Count total occurrences of each category in Category Path
    category_total_occurrences = df_finished['Category Path'].explode().value_counts().reset_index()
    category_total_occurrences.columns = ['Category', 'total_occurrences']

    # Merge total occurrences into back-click data
    category_rating_back_click_counts = pd.merge(category_rating_back_click_counts,category_total_occurrences.rename(columns={'Category': 'back_click_categories'}),on='back_click_categories',how='left')

    # Normalize back-click counts by total occurrences
    category_rating_back_click_counts['normalized_back_clicks'] = (category_rating_back_click_counts['back_click_count']/ category_rating_back_click_counts['total_occurrences'])
    # Reorder both columns (ratings) and rows (categories)
    df_pivot = category_rating_back_click_counts.pivot(index='back_click_categories', columns='rating', values='normalized_back_clicks').fillna(0)
    df_pivot = df_pivot[rating_order_plot2].loc[df_pivot[rating_order_plot2].sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    # ---- Plot 1: Back-Clicks per Rating ----
    back_per_rating = (
        df_finished.groupby("rating", dropna=False)
        .agg(mean_back_clicks=("back_clicks", "mean"), mean_path_length=("path_length", "mean"))
        .reset_index()
    )
    back_per_rating['rating'] = back_per_rating['rating'].fillna('NaN')
    back_per_rating['normalized_back_clicks'] = back_per_rating["mean_back_clicks"] / back_per_rating["mean_path_length"]

    # Plot 1 with shared colors
    sn.barplot(data=back_per_rating,x="rating",hue="rating",y="normalized_back_clicks",palette=[rating_colors[r] for r in rating_order_plot1],ax=ax[0])
    ax[0].set_title("Distribution of Back-Clicks per Rating")
    ax[0].set_xlabel("Rating")
    ax[0].set_ylabel("Mean Back-Clicks number")

    # ---- Plot 2: Stacked bar plot for back-clicks by category and rating ----
    bottom = pd.Series([0] * len(df_pivot), index=df_pivot.index)
    for rating in rating_order_plot2:
        ax[1].bar(df_pivot.index,df_pivot[rating],bottom=bottom,label=f'Rating {rating}',color=rating_colors[rating])
        bottom += df_pivot[rating]

    ax[1].set_title("Normalized Back-Clicks by Category and Rating")
    ax[1].set_xlabel("Category")
    ax[1].set_ylabel("Normalized Back-Clicks")
    ax[1].tick_params(axis='x', rotation=90)

    # ---- Shared Legend ----
    handles = [plt.Rectangle((0, 0), 1, 1, color=rating_colors[rating]) for rating in rating_order_plot1]
    ax[0].legend(handles, [f"{r}" for r in rating_order_plot1], title="Rating", loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_similarity(axs, fig, means, sems, colours, labels):
    """
    Plots the mean name similarity between previous and next articles in user paths.
    
    Parameters:
    - axs (list): List of axes for the subplots.
    - fig (Figure): The plotly figure to annotate.
    - means (list): List of mean similarity values.
    - sems (list): List of standard error of the mean values.
    - colours (list): List of colors for the plot.
    - labels (list): List of labels for the plot.
    """
    for i in range(4):
        sn.lineplot(x=np.arange(1, len(means[i])+1), y=means[i], lw=3, color=colours[i], label=labels[i], ax=axs[i//2])

        axs[i//2].fill_between(
            np.arange(1, len(means[i])+1), 
            np.array(means[i][:10]) - np.array(sems[i][:10]), 
            np.array(means[i][:10]) + np.array(sems[i][:10]), 
            color=colours[i], alpha=0.3)

    axs[0].set_title('BERT embedding')
    axs[0].set_ylabel('Normalised Mean Similarity')
    axs[0].set_xticks(np.arange(1, len(means[0])+1))
    axs[0].set_xlabel('Step Along Path')

    axs[1].set_xticks(np.arange(1, len(means[0])+1))
    axs[1].set_xlabel('Step Along Path')
    axs[1].set_title('BGEM3 embedding')

    plt.suptitle('Mean Name Similarity Between Previous and Next Articles in User Paths')
    plt.tight_layout()
    plt.legend()

    fig.patch.set_facecolor('#fafaf9')  
    axs[0].set_facecolor('w')  
    axs[1].set_facecolor('w')  

    plt.savefig('./figures/similarity.png', bbox_inches='tight', dpi=1000)
    plt.show()