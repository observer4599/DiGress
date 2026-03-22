# DiGress: Discrete Denoising diffusion models for graph generation

Update (Nov 20th, 2023): Working with large graphs (more than 100-200 nodes)? Consider using SparseDiff, a sparse version of DiGress: https://github.com/qym7/SparseDiff

Update (July 11th, 2023): the code now supports multi-gpu. Please update all libraries according to the instructions. 
All datasets should now download automatically

  - For the conditional generation experiments, check the `guidance` branch.
  - If you are training new models from scratch, we recommand to use the `fixed_bug` branch in which some neural
network layers have been fixed. The `fixed_bug` branch has not been evaluated, but should normally perform better.
If you train the `fixed_bug` branch on datasets provided in this code, we would be happy to know the results.

## Environment installation
This code was tested with PyTorch 2.1, Python 3.11, rdkit 2023.03.2 and torch_geometric 2.3.1

The environment is managed with [Pixi](https://pixi.sh/latest/).

  - Install Pixi:

    ```curl -fsSL https://pixi.sh/install.sh | sh```

  - Install the environment:

    ```pixi install```

  - Navigate to the `./src/analysis/orca` directory and compile orca.cpp:

    ```g++ -O2 -std=c++11 -o orca orca.cpp```

### GPU training on Linux with CUDA

Add the following sections to `pixi.toml`, then re-run `pixi install`. Replace `cu118` with your CUDA version (e.g. `cu121`, `cu124`):

```toml
[system-requirements]
cuda = "11.8"

[pypi-options]
extra-index-urls = ["https://download.pytorch.org/whl/cu118"]
```

## Run the code

All commands are run from the project root using `pixi run`. Check [hydra documentation](https://hydra.cc/) for overriding default parameters.

  - To run the debugging code (recommended first): `pixi run python src/main.py +experiment=debug.yaml`
  - To run on only a few batches: `pixi run python src/main.py general.name=test`
  - To run the discrete model (default): `pixi run python src/main.py`
  - To run the continuous model: `pixi run python src/main.py model=continuous`
  - To specify a dataset: `pixi run python src/main.py dataset=guacamol` — see `configs/dataset/` for available datasets

**Important:** The graph benchmark datasets (planar, sbm, comm20) require a matching `+experiment=` flag to load dataset-specific hyperparameters (batch size, model dimensions, etc.). Without it the default `batch_size=512` will cause an out-of-memory error on the edge feature tensors:

  ```bash
  pixi run python src/main.py dataset=planar  +experiment=planar   # planar graphs
  pixi run python src/main.py dataset=sbm     +experiment=sbm      # stochastic block model
  pixi run python src/main.py dataset=comm20  +experiment=comm20   # 20-community graphs
  ```

A `train` task shortcut is also defined in `pixi.toml`:

  ```bash
  pixi run train                        # discrete model (default: QM9)
  pixi run train model=continuous       # continuous model
  pixi run train dataset=guacamol       # different dataset
  ```
    
## Checkpoints

**My drive account has unfortunately been deleted, and I have lost access to the checkpoints. If you happen to have a downloaded checkpoint stored locally, I would be glad if you could send me an email at vignac.clement@gmail.com or raise a Github issue.**

The following checkpoints should work with the latest commit:

  - [QM9 (heavy atoms only)](https://drive.switch.ch/index.php/s/8IhyGE4giIW1wV3) \\
  
  - [Planar](https://drive.switch.ch/index.php/s/8IhyGE4giIW1wV3) \\

  - MOSES (the model in the paper was trained a bit longer than this one): https://drive.google.com/file/d/1LUVzdZQRwyZWWHJFKLsovG9jqkehcHYq/view?usp=sharing -- This checkpoint has been sent to me, but I have not tested it. \\

  - SBM: ~~https://drive.switch.ch/index.php/s/rxWFVQX4Cu4Vq5j~~ \\
    Performance of this checkpoint:
    - Test NLL: 4757.903
    - `{'spectre': 0.0060240439382095445, 'clustering': 0.05020166160905111, 'orbit': 0.04615866844490847, 'sbm_acc': 0.675, 'sampling/frac_unique': 1.0, 'sampling/frac_unique_non_iso': 1.0, 'sampling/frac_unic_non_iso_valid': 0.625, 'sampling/frac_non_iso': 1.0}`

  - Guacamol: https://drive.google.com/file/d/1KHNCnPJmPjIlmhnJh1RAvhmVBssKPqF4/view?usp=sharing -- This checkpoint has been sent to me, but I have not tested it.

## Generated samples

We provide the generated samples for some of the models. If you have retrained a model from scratch for which the samples are
not available yet, we would be very happy if you could send them to us!


## Troubleshooting 

`PermissionError: [Errno 13] Permission denied: '/home/vignac/DiGress/src/analysis/orca/orca'`: You probably did not compile orca.
    

## Use DiGress on a new dataset

To implement a new dataset, you will need to create a new file in the `src/datasets` folder. Depending on whether you are considering
molecules or abstract graphs, you can base this file on `moses_dataset.py` or `spectre_datasets.py`, for example. 
This file should implement a `Dataset` class to process the data (check [PyG documentation](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html)), 
as well as a `DatasetInfos` class that is used to define the noise model and some metrics.

For molecular datasets, you'll need to specify several things in the DatasetInfos:
  - The atom_encoder, which defines the one-hot encoding of the atom types in your dataset
  - The atom_decoder, which is simply the inverse mapping of the atom encoder
  - The atomic weight for each atom atype
  - The most common valency for each atom type

The node counts and the distribution of node types and edge types can be computed automatically using functions from `AbstractDataModule`.

Once the dataset file is written, the code in main.py can be adapted to handle the new dataset, and a new file can be added in `configs/dataset`.


## Cite the paper

```
@inproceedings{
vignac2023digress,
title={DiGress: Discrete Denoising diffusion for graph generation},
author={Clement Vignac and Igor Krawczuk and Antoine Siraudin and Bohan Wang and Volkan Cevher and Pascal Frossard},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=UaAD-Nu86WX}
}
```
