<h1><span style= "color: #c7c9cf">The5Outliers - </span><span style="background: linear-gradient(to right, #3458d6, #34d634); -webkit-background-clip: text; color: transparent;">Wikispeedia Voyages</span><span style= "color: #c7c9cf">: why so many players pass through Geography or Countries to reach their target</span></h1>


<h2 style= "color: #c7c9cf"> Abstract </h2> 
A considerable number of users go through countries or articles about geography to find their target, irrespective of the categories of the initial and target articles: let’s call this the <span style="background: linear-gradient(to right, #3458d6, #34d634); -webkit-background-clip: text; color: transparent;">Wikispeedia Voyage</span>. This raises the question whether human reasoning in the game is inherently tied to countries and geography or whether these articles simply have interesting properties for navigation. Moreover, if humans make <span style="background: linear-gradient(to right, #3458d6, #34d634); -webkit-background-clip: text; color: transparent;">Wikispeedia Voyages</span> because they are easier than finding direct paths, how efficient are they in comparison? To find an answer to these questions, ... quick overview of methods. Maybe analyse the semantic of the categories in parsed articles?


Initial analyses showed that articles about countries and geography indeed have a prominent place in the <i>Wikispeedia</i> network. The analysis is based on several features: 

Through player behaviour we will show that country or geography articles . However, players perhaps struggle less going through these intuitive categories to link start and target. The . Finally, to characterise the efficiency of <span style="background: linear-gradient(to right, #3458d6, #34d634); -webkit-background-clip: text; color: transparent;">Wikispeedia Voyages</span>, we compare them to other paths: do article nodes have higher semantic similarity, are the trajectories more direct, the games longer or more likely to be abandoned? 


<h2 style= "color: #c7c9cf"> Research questions </h2> 
<ol> 
    <li> Do geography and country articles simply have interesting properties that make them particularly useful in the game? For instance, are they more numerous, more connected or have higher link density? </li>
    <li> Are paths going through geography and country categories efficient or are they rather an easy but long way to reach the goal? How do they compare to the optimal path or paths going through other categories? </li>
    <li> Is there a semantic detour taken through <span style="background: linear-gradient(to right, #3458d6, #34d634); -webkit-background-clip: text; color: transparent;">Wikispeedia Voyages</span>? That is, are articles successions on more direct paths semantically closer?  </li>
    <li> Are countries inherently tied to how humans reason about the <i>Wikispeedia</i> route they envision? Can we find evidence or hints in the user behaviours that they are more comfortable with certain categories? </li>
    <li> Some other questions? </li>
    
</ol>

<h2 style= "color: #c7c9cf"> Methods </h2> 
<h3 style= "color: #c7c9cf"> Semantic similarity </h3>

The semantic similarity matrices are computed in a few different ways. One way is to compute them directly  through the article names using BGEM3<sup>1</sup> as embedding model. The similarity between two articles with embedded name vectors a<sub>1</sub> and a<sub>2</sub> is defined as the cosine similarity 
$$S_C(a_1, a_2)=\frac{a_1 \cdot a_2}{||a_1|| \cdot ||a_2||}$$
Another way to define semantic similarity is through the categories using Jaccard similarity.

<h3 style= "color: #c7c9cf"> html parsing </h3>
Parsing allows to find interesting features of the wikipedia articles: the number of words and links, and hence the link density. It also gives structural information about the pages: categories, subcategories, the presence and nature of tables. For each of these structures, the number of words and the list of present links is reported. 





high link density or be particularly numerous, giving them a central position in the navigation network

difficulty perceived by players is measured through a combination of parameters: the duration and path length of their game, the number of back-clicks needed, the difficulty rating given, whether games were finished and if not, how they were abandoned

<h2 style= "color: #c7c9cf"> Timeline and organisation </h2> 
Week 1:

Week 2:

Week 3:

Week 4:

Week 5:


<h2 style= "color: #c7c9cf"> References </h2> 
[1] Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. <i>arXiv</i>, 2024.

<h3 style= "color: #c7c9cf"> Quickstart </h3> 

```bash
# clone project
git clone <project link>
cd <project repo>

# create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>


# install requirements
pip install -r pip_requirements.txt
```



<h3 style= "color: #c7c9cf"> How to use the library </h3> 
Tell us how the code is arranged, any explanations goes here.



<h3 style= "color: #c7c9cf"> Project Structure </h3> 

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```
