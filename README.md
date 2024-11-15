# The5Outliers - *Wikispeedia Voyages*: why so many players pass through Geography or Countries to reach their target


<h2 style= "color: #c7c9cf"> Abstract </h2> 

A considerable number of users go through countries or articles about geography to find their target, irrespective of the categories of the initial and target articles: let’s call this the *Wikispeedia Voyage*. This raises the question whether human reasoning in the game is inherently tied to countries and geography or whether these articles simply have interesting properties for navigation. Moreover, if humans make *Wikispeedia Voyages* because they are easier than finding direct paths, how efficient are they in comparison? To answer these questions, we will start with an analysis of the key factors involved: article features (link density, connectivity), player behaviour (game duration and path length, number of back clicks, etc.) and category caracteristics (centrality, flows between categories). Then, we aim to find correlations between above mentionned key factors and compare particularities of different clusters found for instance based on categories. Finally, we formally define *Wikispeedia Voyages* and thoroughly analyse the semantic, behavioural and performance aspects of this strategy.



<h2 style= "color: #c7c9cf"> Research questions </h2> 

1. Do certain articles have interesting properties that make them particularly useful in the game? For instance, are they more numerous, more connected or have higher link density? 
2. What can we say about the prominence or centrality of different categories of articles?
3. How can we rigorously define *Wikispeedia Voyages*?
4. Are paths going through geography and country categories efficient? Are they rather an easy but long way to reach the goal? How do they compare to the optimal path or paths going through other categories? 
5. Is there a semantic detour taken through *Wikispeedia Voyages*? That is, are articles successions on more direct paths semantically closer? 
6. Are countries inherently tied to how humans reason about the *Wikispeedia* route they envision? Can we find evidence or hints in the user behaviours that they are more comfortable with certain categories? 

<h2 style= "color: #c7c9cf"> Methods </h2> 
<h3 style= "color: #c7c9cf"> Semantic similarity </h3>

The semantic similarity matrices are computed in a few different ways. One way is to compute them directly through the article names using BGEM3<sup>1</sup> as embedding model. The similarity between two articles with embedded name vectors a<sub>1</sub> and a<sub>2</sub> is defined as the cosine similarity 

$$S_C(a_1, a_2)=\frac{a_1 \cdot a_2}{||a_1|| \cdot ||a_2||}$$

Another way to define semantic similarity is through the categories using Jaccard similarity. Jaccard similarity assigns a score based on the intersection of categories and subcategories of an article. For instance, two articles A and B that have the same category will be assigned a score of 1. If they have also the same subcategory, the score will raise to 3. If there is no overlap, similarity is 0.

$$J(A, B)=\frac{|A \cap B|}{|A \cup B|}$$


<h3 style= "color: #c7c9cf"> HTML parsing </h3>
Parsing allows to find interesting features of the wikipedia articles: the number of words and links, and hence the link density. It also gives structural information about the pages: categories, subcategories, the presence and nature of tables. For each of these structures, the number of words and the list of present links is reported. 

<h3 style= "color: #c7c9cf"> Networks and Graph </h3>

Networks are particularly useful for representation of complex relations between for example categories or category transitions.

- **Nodes:** Each node can represents a group of articles belonging to a specific category, with each article assigned to his main category. 
- **Edges:** Each edge can represents a transition from one article category to another, indicating the flow of user activity.

The node positions are determined using a **force-directed algorithm** where each *nodes* repel each other like charged particles (sometimes called an anti-gravity force) preventing overlaps and the *edges* act like springs, pulling connected nodes closer to reflect their relationships.

Furthermore, they are used as clustering for the similarity measures: reordering of the articles based on similarity allows to find meaningful clusters of similar articles.

<h3 style= "color: #c7c9cf"> Assignment of methods to research questions </h3>

1. Statistical analysis of article features. 
2. Networks: centrality based on the connections between articles in certain categories and centrality based on the flow between categories from user paths.
3. Aggregate categories and check if they pass through geography or country.
4. Compare key factors for the voyage and non-voyage paths: for subsets of similar paths (e.g. same start and end category), compare the difficulty experiences by the player, the duration and length of the game and the ideal path. 
5. Includes similarity analysis on paths (eventually with interpolation to compare paths of different lengths), showing whether users zoom out more from a semantic cluser in voyages.
6. Explore different possible confounfing factors between human *navigation* and human *reasoning*, evaluate if users pass through geography and countries significantly more than needed and more than other categories, considering the optimal paths.


<h2 style= "color: #c7c9cf"> Additional Datasets </h2> 

No additional datasets are needed to answer the research questions.

<h2 style= "color: #c7c9cf"> Timeline </h2> 

Step 1 (due to 24.11): Core analysis on the voyage vs non-voyage categories, including correlations with difficulty metrics and similarity analysis.

Step 2 (due to 01.12): In-depth compairison between user path and optimal path, look for significant differences in geography and country categories.

Step 3 (due to 08.12): Selection of pertinent results for the website, create interactive plots and redaction of the story.

Step 4 (due to 20.12): Build the website.

<h2 style= "color: #c7c9cf"> Organization within the team </h2> 

Our team of five works collaboratively, contributing equally to the project by pairing up for tasks to ensure thorough double-checking, and rotating pairs when switching tasks. In the end, each of us actively participates in both the analysis and the building of the website.

<h2 style= "color: #c7c9cf"> References </h2> 
[1] Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. <i>arXiv</i>, 2024.

<h3 style= "color: #c7c9cf"> Quickstart </h3> 

```bash
# clone project
git clone https://github.com/epfl-ada/ada-2024-project-the5outliers.git

# create conda called 'ada_project' with all required packages
conda env create -f requirements.yml
```



<h3 style= "color: #c7c9cf"> How to use the library </h3> 

All the results are in the ```results.ipynb```. Running the notebook will showcase the different functionalities and models defined under src.



<h3 style= "color: #c7c9cf"> Project Structure </h3> 

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── requirements.txt            <- File for installing python dependencies
└── README.md
```
