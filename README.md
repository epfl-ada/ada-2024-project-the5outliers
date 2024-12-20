# The5Outliers - *Wikispeedia Voyages*: Why so many people use World Regions to reach their target.

You can look at our website [here](https://yannickdetrois.github.io/epfl-ada-datastory/).

<h2 style= "color: #c7c9cf"> Abstract </h2> 

While playing [Wikispeedia](https://dlab.epfl.ch/wikispeedia/play/), we observed that we tend to adopt a strategy of navigating through articles about World Regions to reach the target, an approach we term *Wikispeedia Voyages*. A Voyage is defined as a path where neither the source nor target is in the World Regions category, but the path includes at least one article from this category. The Wikispeedia [dataset](https://snap.stanford.edu/data/wikispeedia.html) collects both information about players' behaviour and the network structure of Wikipedia articles. Our aim is to understand to what extent the article and network structure, analyzed through Markov Chains, influences gameplay and the choice to undertake Voyages. In parallel, user difficulty in Voyages are also compared to other strategies, their alignment with shortest paths, as well as their insights from semantic similarity of article names along the paths.

Our results show that World Regions articles are highly connected, with a dense network of links. Moreover, a Markov Chains analysis showed that users navigate through this category more frequently than random walks in the network could suggest. Interestingly, a comparison of user paths with optimal (shortest) paths reveals that optimal paths leverage World Regions more often than users, suggesting that Voyages are effective strategies that players might underuse. Users achieve a higher success rate in reaching their targets when employing *Wikispeedia Voyages*, but seem to take longer and need more back-clicks, which could be due to the lesser semantic similarity observed throughout *Voyages*. However, only a small subset of articles within World Regions plays a particularly significant role in facilitating successful navigation, inviting to rather consider only larger countries, continents or regions for *Voyages*.

<h2 style= "color: #c7c9cf"> Research questions </h2> 

1. Does the Wikispeedia article and network structure intrinsically favour *Wikispeedia Voyages*? For example, are World Regions more numerous or more connected? Does the page structure of articles have an influence on *Wikispeedia Voyages*?
2. Are users faster or more efficient when taking Wikispeedia Voyages, or do they take semantic detours that could complicate the path? 
3. How does the strategy compare with the algorithmic shortest paths?

<h2 style= "color: #c7c9cf"> Methods </h2> 

<h3 style= "color: #c7c9cf"> HTML parsing </h3>
Parsing allows to find interesting features of the wikipedia articles: the number of words and the total number of links, taking in acount duplicates inside the same page. It also gives structural information about the pages: categories, subcategories, the presence and nature of tables. For each of these structures, the number of words and the list of present links is reported. 

<h3 style= "color: #c7c9cf"> Networks and Graphs </h3>
Optimal (shortest) paths are determined by modeling the Wikispeedia article network as a directed graph, where nodes represent articles and edges represent hyperlinks between them. For each source-target pair present in the users' games, we compute all shortest paths, ensuring that every minimum-length route is considered.

<h3 style= "color: #c7c9cf"> Markov Chains </h3>


Markov Chains are used to model the influence of the network structure that could inherently bias user paths. Every article is assigned transition probabilities to all other articles based on the number of links present in this article. The transition matrix's $Q$ normalised left eigenvector with eigenvalue $1$ (i.e. $x$ such that $xQ=x$) gives the steady-state of the system (useful to find central articles of the network). 

To compare with the user paths, we can count the number of transitions at every step and regroup them in a matrix P. To see the deviation with the random path we use the Kullback–Leibler ($KL$) divergence, defined element-wise as 

$$
D_{KL}(P || Q)\_{ij} = P_{ij} \log \frac{P_{ij}}{Q_{ij}}
$$


if $P_{ij} > 0$ and $Q_{ij} > 0$ and $0$ otherwise.


<h3 style= "color: #c7c9cf"> Difficulty Metrics </h3>
The distributions of game duration, path length, back-clicks and ratings for both Voyages and non-Voyages are extracted and compared. The proportion of finished paths is also calculated and compared.



<h3 style= "color: #c7c9cf"> Semantic similarity </h3>

The semantic similarity matrices are computed in a few different ways. One way is to compute them directly through the article names using BGEM3<sup>1</sup> and BERT as embedding model. The similarity between two articles with embedded name vectors a<sub>1</sub> and a<sub>2</sub> is defined as the cosine similarity. 

$$S_C(a_1, a_2)=\frac{a_1 \cdot a_2}{||a_1|| \cdot ||a_2||}$$

<h3 style= "color: #c7c9cf"> Assignment of methods to research questions </h3>

1. Leverage features from HTML parsing and Markov Chains to evaluate article connectivity, transition probabilities, and the influence of link positions on user choices.
2. Compare difficulty metrics and success rates between *Voyages* and other strategies. Use semantic similarity analysis to assess whether *Voyages* exhibit lower cosine similarity between steps compared to other paths taken by users.
3. Construct a directed graph to compute optimal paths, and calculate the normalized percentage of times each category is visited.

<h2 style= "color: #c7c9cf"> Additional Datasets </h2> 

No additional datasets are needed to answer the research questions.

<h2 style= "color: #c7c9cf"> Contributions </h2> 

Our team of five collaborated on the initial exploratory analysis, identifying research questions, and drafting the data story. We then divided specific tasks more concretely:

Camille Challier: Difficulty metrics, page structure analysis, and random path comparisons.

Yannick Detrois: HTMLParser, Website Design and Redaction, Markov Chains, Similarity along paths

David Friou: Data preprocessing, handling category articles, working on networks, and proper connection and functionality of the entire code.

Marine Ract: User Networks, Markov chains, Sankey plots, Website Design.

Marianne Scoglio: Extraction of Voyages, comparaison between user and optimal paths. 

<h2 style= "color: #c7c9cf"> References </h2> 
[1] Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. <i>arXiv</i>, 2024.

<h2 style= "color: #c7c9cf"> Quickstart </h3> 

```bash
# clone project
git clone https://github.com/epfl-ada/ada-2024-project-the5outliers.git

# create conda called 'ada_p' with all required packages
conda env create -f requirements.yml
```

<h3 style= "color: #c7c9cf"> How to use the library </h3> 

All the results are in the ```results.ipynb```. Running the notebook will showcase the different functionalities and models defined under src.



<h3 style= "color: #c7c9cf"> Project Structure </h3> 

The directory structure of our project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│
├── results.ipynb               <- Our Notebook showing our main results 
│
├── .gitignore                  <- List of files ignored by git
├── requirements.yml            <- File for installing python dependencies
└── README.md
```
