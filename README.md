[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Toward Graph Data Collaboration in a Data-Sharing-Free Manner: A Novel Privacy-Preserving Graph Pre-training Model

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper 
[Toward Graph Data Collaboration in a Data-Sharing-Free Manner: A Novel Privacy-Preserving Graph Pre-training Model](https://doi.org/10.1287/ijoc.2023.0115) by Jiarong Xu, Jiaan Wang, Zenan Zhou and Tian Lu.


# Cite
To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2023.0115

https://doi.org/10.1287/ijoc.2023.0115.cd

Below is the BibTex for citing this snapshot of the repository.

```
@misc{Jiarong2025Toward,
  author =        {Xu, Jiarong and Wang, Jiaan and Zhou, Zenan and Lu, Tian},
  publisher =     {INFORMS Journal on Computing},
  title =         {{Toward Graph Data Collaboration in a Data-Sharing-Free Manner: A Novel Privacy-Preserving Graph Pre-training Model}},
  year =          {2025},
  doi =           {10.1287/ijoc.2023.0115.cd},
  url =           {https://github.com/INFORMSJoC/2023.0115},
  note =          {Available for download at https://github.com/INFORMSJoC/2023.0115},
}  
```

# Data

We use eight datasets in the experiments: Deezer, Facebook, LastFM, DBLP, Amazon, Twitter, Twitter-Foursquare and Phone-Email.

Among them, we provide the data files of Deezer, Facebook, LastFM, Twitter-Foursquare and Phone-Email in the `data/` folder. There datasets are all open-source datasets, we also list their original sources:
- Deezer: https://snap.stanford.edu/data/feather-deezer-social.html
- Facebook: https://snap.stanford.edu/data/facebook-large-page-page-network.html
- LastFM: https://snap.stanford.edu/data/feather-lastfm-social.html
- Twitter-Foursquare: https://github.com/zhichenz98/PARROT-WWW23/tree/master/datasets
- Phone-Email:https://github.com/zhichenz98/PARROT-WWW23/tree/master/datasets

For DBLP, Amazon and Twitter, due to the large datasize, we only list their original sources:
- DBLP: https://snap.stanford.edu/data/com-DBLP.html
- Amazon: https://snap.stanford.edu/data/amazon-meta.html
- Twitter: https://snap.stanford.edu/data/ego-Twitter.html


# Replication

## Environment 

You should install the following packages in the environment:
- python >= 3.8
- torch >= 1.11.0
- dgl==0.4.3
- torch_geometric==2.0.4
scikit-learn==0.20.3
scipy==1.4.1
coverage==4.5.4
coveralls==1.9.2
black==19.3b0
pytest==5.3.2
networkx==2.3
numpy==1.18.2
tensorboard_logger==0.1.0

## Scripts

### data processing
The data processing script is provided in `scripts/generate_pretraining_and_downstream_dataset.py`. For each dataset, we use this script to generate the pretraining and downstream graphs. The run command is shown as follows:

```python
python -u scripts/generate_pretraining_and_downstream_dataset.py --dataset [dateset_name]
```

### Model Structure
The structure of GIN network and graph encoder are provided in `scripts/gin_edge_weighted.py` (class GIN_Edge_Weighted) and `scripts/graph_encoder_edge_weighted.py` (class GraphEncoder_Edge_Weighted).

### Downstream Tasks

To run link prediction task:
```
bash scripts/predict_links_of_pretraining_dataset.sh <gpu> <load_path> <hidden_size> <dowstream_dataset> <pretraining_dataset>
```

Tu run node classification task:
```
bash scripts/node_classification/classify_node_of_downstream_dataset.sh <gpu> <load_path> <hidden_size> <downstream_dataset>
```


### Baselines
- For the implementation of GAL-W and GAL-TV, please refer to [GAL](https://github.com/liaopeiyuan/GAL).
- For the implementation of EdgeRand and LapGraph, please refer to [LinkTeller](https://github.com/AI-secure/LinkTeller).
- For the implementation of GCC, please refer to [GCC](https://github.com/THUDM/GCC)



# Acknowledgements

Part of this code is inspired by [GCC](https://github.com/THUDM/GCC) and [CMC](https://github.com/HobbitLong/CMC)


