# The5Outliers - *Wikispeedia Voyages*: Why so many people use World Regions to reach their target.


<h2 style= "color: #c7c9cf"> Abstract </h2> 

In playing [Wikispeedia](https://dlab.epfl.ch/wikispeedia/play/), we observed a common strategy of navigating through articles about World Regions to reach the target, an approach we term *Wikispeedia Voyages*. A Voyage is defined as a path where neither the source nor target is in the World Regions category, but the path includes at least one article from this category. The Wikispeedia [dataset](https://snap.stanford.edu/data/wikispeedia.html) collects both information about players' behaviour and the network structure of Wikipedia articles. Our aim is to understand to what extent the network structure, analyzed through Markov Chains, influences gameplay and the choice to undertake Voyages, while also examining user difficulty in Voyages compared to other strategies, their alignment with shortest paths, and insights from semantic analysis.

Our results show that World Regions articles are highly connected, with a dense network of inbound links. Moreover, a Markov Chains analysis showed that users navigate through this category more frequently than random algorithms do. Interestingly, a comparison of user paths with optimal (shortest) paths reveals that optimal paths leverage World Regions more often than users, suggesting that Voyages are effective strategies that players might underuse. Users achieve a higher success rate in reaching their targets when employing *Wikispeedia Voyages*. However, a small subset of articles within World Regions plays a particularly significant role in facilitating successful navigation.

<h2 style= "color: #c7c9cf"> Research questions </h2> 

1. Does the Wikispeedia article network structure intrinsically favour *Wikispeedia Voyages*? For example, are World Regions more numerous or more connected? And does the page structure have an influence on *Wikispeedia Voyages*?
2. Are users more comfortable, fast or efficient with World Regions than other topics? Is there a semantic detour taken through *Wikispeedia Voyages*?
3. How does the strategy compare with the algorithmic shortest paths?

<h2 style= "color: #c7c9cf"> Methods </h2> 

<h3 style= "color: #c7c9cf"> HTML parsing </h3>
Parsing allows to find interesting features of the wikipedia articles: the number of words and the total number of links,taking in acount duplicates inside the same page. It also gives structural information about the pages: categories, subcategories, the presence and nature of tables. For each of these structures, the number of words and the list of present links is reported. 

<h3 style= "color: #c7c9cf"> Networks and Graphs </h3>

TODO: Markov Chains

Optimal (shortest) paths are determined by modeling the Wikispeedia article network as a directed graph, where nodes represent articles and edges represent hyperlinks between them. For each source-target pair present in the users' games, we compute all shortest paths, ensuring that every minimum-length route is considered.

<h3 style= "color: #c7c9cf"> Semantic similarity </h3>

The semantic similarity matrices are computed in a few different ways. One way is to compute them directly through the article names using BGEM3<sup>1</sup> and BERT as embedding model. The similarity between two articles with embedded name vectors a<sub>1</sub> and a<sub>2</sub> is defined as the cosine similarity 

$$S_C(a_1, a_2)=\frac{a_1 \cdot a_2}{||a_1|| \cdot ||a_2||}$$

<h3 style= "color: #c7c9cf"> Assignment of methods to research questions </h3>

1. Leverage features from HTML parsing and Markov Chains to evaluate article connectivity, transition probabilities, and the influence of link positions on user choices.
2. Compare difficulty metrics (such as path length, duration, rating, and back-clicks) and success rates between *Voyages* and other strategies. Use semantic similarity analysis to assess whether steps towards *World Regions* exhibit lower cosine similarity compared to other steps taken by users.
3. Construct a directed graph to compute optimal paths, and calculate the normalized percentage of times each category is visited.

<h2 style= "color: #c7c9cf"> Additional Datasets </h2> 

No additional datasets are needed to answer the research questions.

<h2 style= "color: #c7c9cf"> Contributions </h2> 

Our team of five collaborated on the initial exploratory analysis, identifying research questions, and drafting the data story. We then divided specific tasks more concretely:

Camille Challier:

Yannick Detrois: 

David Friou: Data preprocessing, handling category articles, working on networks, and proper connection and functionality of the entire code.

Marine Ract: 

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
