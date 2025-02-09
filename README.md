# Genifer (generative feature-driven image replay)

Generative feature-driven image replay (Genifer) is a class-incremental learning method that does not require access to any previously seen samples. A generative model is trained to replay images that must induce the same hidden features as real samples when they are passed through the classifier. This repository contains the PyTorch code of our corresponding paper [Generative feature-driven image replay for continual learning](https://www.sciencedirect.com/science/article/abs/pii/S0262885624002920).

## Set up environment
Create a conda environment and install the required packages from the provided [environment.yml](./environment.yml):

```
git clone https://github.com/kevthan/feature-driven-image-replay
cd feature-driven-image-replay
conda env create -f environment.yml
conda activate genifer
```

## Set up CUDA

Go to [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) and download the correct toolkit depending on your OS, architecture etc.

Example for linux:

```
chmod +x ...

./cuda_12.6.1_560.35.03_linux.run --tmpdir=./tmp/ --installpath=./

Driver:   Not Selected
Toolkit:  Installed in /your/installpath/

export PATH=/your/installpath/bin:$PATH
export LD_LIBRARY_PATH=/your/installpath/lib64:$LD_LIBRARY_PATH
```

## Integrate external sources

### ADA
Clone the ADA repository for adaptive GAN augmentations to a separate directory outside this one

```
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git /path/to/your/ada
```

Create a symlink called `augmentation` to the ADA code in the subdirectory under [genifer/](genifer/):

```
cd /path/to/feature-driven-image-replay/genifer/
ln -s /path/to/your/ada augmentation
cd ..
```

Add it to the python path:

```
export PYTHONPATH=/path/to/feature-driven-image-replay/genifer/augmentation/:$PYTHONPATH
```

### RAdam Optmizer
Download https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam/radam.py to [genifer/utils/](genifer/utils/).

## Model training & evaluation

Before starting the training, download the CIFAR100 dataset:

```
python bin/download_datasets.py --dataset_path /path/to/your/genifer_datasets
```

In this example, the 100 classes of CIFAR100 are split into an initial task with 50 classes followed by 5 tasks with 10 classes each.

### Classifier pretraining

Make sure to specify all required paths, e.g., `dataset_path`, `results_path`, `classifier_chpt` etc. in the corresponding config files.

Train the classifier on the first task (50 classes):

```
python bin/pretrain.py --config_path genifer/config/examples/pretrain_classifier.json --pretrain_mode classifier
```

### GAN pretraining

Train the GAN for feature-driven image replay on the first task (based on the classifier from the step above):

```
python bin/pretrain.py --config_path genifer/config/examples/pretrain_gan.json --pretrain_mode GAN --classifier_chpt path/to/classifier/results_path/genifer_model_best_val_loss.pt
```

### Tasks t > 1

Specify all required (checkpoint) paths in [genifer/config/examples/incremental_training.json](genifer/config/examples/incremental_training.json). Then start training the GAN and classifier in an alternating manner for all 5 remaining tasks:

```
python bin/train_incremental.py --config_path genifer/config/examples/incremental_training.json --id 123 --start_training_with classifier --start_task 1 --prev_task 0 --end_task 5
```

## Citation
If you use this code, please make sure to cite our work:

```
@article{thandiackal2024genifer,
    title = {Generative feature-driven image replay for continual learning},
    journal = {Image and Vision Computing},
    volume = {150},
    pages = {105187},
    year = {2024},
    author = {Kevin Thandiackal and Tiziano Portenier and Andrea Giovannini and Maria Gabrani and Orcun Goksel},
}
```
