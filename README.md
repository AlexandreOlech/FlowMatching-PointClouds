## Requirements
- python 3.10.18
- pytorch 2.1.0+cu121
- pytorch-lightning 2.5.5
- polyscope 2.4.0
- tensorboard 2.20.0

## Download Data

- Download the ShapeNet dataset using this link:  
  [https://drive.google.com/drive/folders/1MMRp7mMvRj8-tORDaGTJvrAeCMYTWU2j?usp=sharing](https://drive.google.com/drive/folders/1MMRp7mMvRj8-tORDaGTJvrAeCMYTWU2j?usp=sharing)

## Preprocess the Dataset

- Specify the parent folder where you stored the ShapeNet dataset in the `preprocess` section of `configs/default.yaml`.
- Run:
  ```bash
  python preprocess.py
  ```

## Training

- Specify a category in the `train` section of `configs/default.yaml`.  
  You can look at `categ_to_id.json` to see all available categories.
- Specify the location of the preprocessed dataset in `configs/default.yaml`.
- Run:
  ```bash
  python train.py
  ```
- You can use the following command to visualize training and validation losses.
  ```bash
  tensorboard --logdir logs
  ```

## Sampling & Visualization

- In `sample.py`, specify the path to the checkpoint of the trained model. By default, checkpoints of trained models are stored in the `logs` folder. You can also use the pre-trained checkpoint at the root level of the repository.
- Run:
  ```bash
  python sample.py
  ```
  It will display a polyscope interface, allowing you to visualize multiple sampled point clouds.
