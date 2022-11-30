# Data Provenance Inference in Machine Learning
The prototype code for the paper [Data Provenance Inference in Machine Learning](https://arxiv.org/abs/2211.13416).
This repository is targeting for `mobile user` as the data provenance in [OpenImage](https://storage.googleapis.com/openimages/web/index.html) dataset.
## Quick Start
To test the function of this repository, simply run
```python
python script/oi_user_tiny.py
```
The intermediate and final results are saved in `log/res/oi/user_tiny/`.
## Customize Configurations
All the configuration files are in `config/`. The entry configuration file is `config/*.yaml` (e.g. config/oi_user_tiny.yaml) to redirect to the other configuration files for different functional modules.

There are four functional modules in this repository:
- **dataset**: how to extract the raw data of data provenance from the original dataset
- **metadata**: how to split the extracted raw data to facilitate the shadow training
- **model**: the details about how to train the target model and shadow model
- **infer**: the details about how to train and test the meta model for the final data provenance inference

Change the information in the `config/*/*.yaml` (e.g. config/dataset/oi_user_tiny.yaml) to customize any of the above modules' parameters.

**Note**: The current save directory is `data/`, where the raw data, the metadata and the models (DNNs and meta models) are saved. If you want to reorganize the save directories, change the values with the key suffixed with `path`, `dir` or `csv` in `config/*/*.yaml`.

## Contact
If you have any questions about this repository or the paper, please don't hesitate to contact the repository owner or ping <m.xu21@imperial.ac.uk>.

## Citation
If you would like to cite this work, please use the following information:
```text
@article{
        prov-infer,
        author = {{Xu}, Mingxue and {Li}, Xiang-Yang},
        title = "{Data Provenance Inference in Machine Learning,
        journal = {arXiv e-prints},
        year = 2022,
        month = nov,
        eid = {arXiv:2211.13416},
        pages = {arXiv:2211.13416},
        archivePrefix = {arXiv},
        eprint = {2211.13416}
}
```


