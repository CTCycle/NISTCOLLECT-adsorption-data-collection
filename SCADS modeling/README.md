# NISTADS: AI-driven Adsorption Prediction

## 1. Project Overview
This project delves into the fascinating world of adsorption, a surface-based process where a film of particles (adsorbate) accumulate on the surface of a material (adsorbent). This phenomenon plays a pivotal role in numerous industries. For instance, it is widely applied in water treatment facilities for the purification of water, in air filters to improve air quality, and in the automotive industry within catalytic converters to reduce harmful emissions. The adsorption of compounds is usually quantified by measuring the adsorption isotherm a given adsorbate/adsorbent combination. The objective of this project is to harness the power of machine learning to predict the adsorption of chemicals on adsorbent materials. The aim is to build a model that can accurately predict the adsorbed amount of a specific guest-host combination under various conditions, by leveraging the data from the NIST/ARPA-E Database. This could have significant implications for industries that rely on these materials, potentially leading to more efficient processes and better materials design. As such, this project takes a different approach compared to fitting adsorption data with theoretical model for adsorption constants calculation, instead proposing the use of a deep learning approach to understand adsorption isotherm patterns by leveraging a large volume of experimental data.

## 2. Source Data

### 2.1 NIST/ARPA-E Database
The NIST/ARPA-E Database of Novel and Emerging Adsorbent Materials is a free, web-based catalog of adsorbent materials and measured adsorption properties of numerous materials obtained from article entries from the scientific literature. Search fields for the database include adsorbent material, adsorbate gas, experimental conditions (pressure, temperature), and bibliographic information (author, title, journal), and results from queries are provided as a list of articles matching the search parameters. The database also contains adsorption isotherms digitized from the cataloged articles, which can be compared visually online in the web application or exported for offline analytics. 

### 2.2 Adsorption isotherms dataset
The data used for the model training has been extracted as reported in related GitHub repository, available at https://github.com/CTCycle/NISTADS-data-collection. This script makes use of the NIST/ARPA-E Database API to collect adsorption isotherm data for both single component experiments and binary mixture experiments, and conveniently splitting those in two different datasets (Single Component ADSorption and Binary Mixture ADSorption). Since NISTADS is focused on predicting single component adsorption isotherms, it will make use of the single component dataset (SCADS) for the model training.

### 2.1 Data preprocessing
The extracted data undergoes preprocessing via a tailored pipeline, which starts with filtering out undesired experiments. These include experiments featuring negative values for temperature, pressure, and uptake, or those falling outside predefined boundaries for pressure and uptake ranges (refer to configurations for specifics). Pressure and uptakes are standardized to a uniform unit—Pascal for pressure and mol/g for uptakes. Following this refinement, the physicochemical attributes of the absorbate species are unearthed through the PUBCHEM API. This enriches the input data with molecular properties such as molecular weight, the count of heavy atoms, covalent units, and H-donor/acceptor statistics. Subsequently, features pertaining to experiment conditions (e.g., temperature) and adsorbate species (physicochemical properties) are normalized. Names of adsorbents and adsorbates are encoded into integer indices for subsequent vectorization by the designated embedding head of the model. Pressure and uptake series are also normalized, utilizing upper boundaries as the normalization ceiling. Additionally, initial zero measurements in pressure and uptakes series are removed to mitigate potential bias towards zero values. Finally, all sequences are reshaped to have the same length using post-padding with a specified padding value (defaulting to -1 to avoid conflicts with actual values) and then normalized.

## 4. Installation
The installation process is designed for simplicity, using .bat scripts to automatically create a virtual environment with all necessary dependencies. Please ensure that Anaconda or Miniconda is installed on your system before proceeding.

- To set up a CPU-only environment, run `scripts/create_cpu_environment.bat`. This script installs the base version of TensorFlow, which is lighter and does not include CUDA libraries.
- For GPU support, which is necessary for model training on a GPU, use `scripts/create_gpu_environment.bat`. This script includes all required CUDA dependencies to enable GPU utilization.
- Once the environment has been created, run `scripts/package_setup.bat` to install the app package locally.
- **IMPORTANT:** run `scripts/package_setup.bat` if you move the project folder somewhere else after installation, or the app won't work! 

### 4.1 Additional Package for XLA Acceleration
XLA is designed to optimize computations for speed and efficiency, particularly beneficial when working with TensorFlow and other machine learning frameworks that support XLA. By incorporating XLA acceleration, you can achieve significant performance improvements in numerical computations, especially for large-scale machine learning models. XLA integration is directly available in TensorFlow but may require enabling specific settings or flags. To enable XLA acceleration globally across your system, you need to set an environment variable named `XLA_FLAGS`. The value of this variable should be `--xla_gpu_cuda_data_dir=path\to\XLA`, where `path\to\XLA` must be replaced with the actual directory path that leads to the folder containing the nvvm subdirectory. It is crucial that this path directs to the location where the file `libdevice.10.bc` resides, as this file is essential for the optimal functioning of XLA. This setup ensures that XLA can efficiently interface with the necessary CUDA components for GPU acceleration.

## 4. How to use
The user can navigate the project folder to find different subfolders, depending on the desired task to perform. 

**Data:** this folder contains the data utilized for the model training. 
- `adsorbates_dataset.csv` provides information about the adsorbates species;
- `adsorbents_dataset.csv` provides information about the adsorbent materials;
- `SCADS_dataset.csv` contains the data that will be used for SCADS training;
Run `data_validation.ipynb` to start a jupyter notebook for explorative data analysis (EDA) of the adsorption isotherm dataset.

**Training:** contains the necessary files for conducting model training and evaluation. `SCADS/model/checkpoints` acts as the default repository where checkpoints of pre-trained models are stored. Run `model_training.py` to initiate the training process for deep learning models, or launch `model_evaluation.ipynb` to perform a model performance analysis using different metrics on pretrained models checkpoints. 

**Inference:** use `adsorption_predictions.py` from this directory to predict adsorption given empirical pressure series and parameters. The predicted values are saved in `SCADS_predictions.py`. 

### 4.1. Configurations
The configurations.py file allows to change the script configuration. 

| Category              | Setting               | Description                                                              |
|-----------------------|-----------------------|--------------------------------------------------------------------------|
| **Advanced settings** | use_mixed_precision   | Whether to use mixed precision for faster training (mix float16/float32).|
|                       | use_tensorboard       | Activate or deactivate tensorboard logging.                              |
|                       | XLA_acceleration      | Use of linear algebra acceleration for faster training.                  |
|                       | training_device       | Select the training device (CPU or GPU).                                 |
|                       | num_processors        | Number of processors (cores) to be used during training.                 |
| **Training settings** | epochs                | Number of training iterations.                                           |
|                       | learning_rate         | Learning rate of the model.                                              |
|                       | batch_size            | Size of batches to be fed to the model during training.                  |
| **Model settings**    | embedding_dims        | Embedding dimensions for guest and host inputs.                          |
|                       | generate_model_graph  | Generate and save 2D model graph (as .png file).                         |
| **Training data**     | num_samples           | Number of experiments to consider for training.                          |
|                       | test_size             | Size of the test dataset (fraction of total samples).                    |
|                       | pad_length            | Max length of the pressure/uptake series (for padding).                  |
|                       | pad_value             | Number to assign to padded values (default: -1).                         |
| **General settings**  | seed                  | Global random seed.                                                      |
|                       | split_seed            | Random seed for dataset split.                                           |

## License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.
