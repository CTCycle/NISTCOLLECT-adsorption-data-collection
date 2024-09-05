# NISTADS: NIST/ARPA-E dataset composer and adsorption modeling

## 1. Project Overview
The NISTADS project delves into the fascinating world of adsorption, a surface-based process where a film of particles (adsorbate) accumulate on the surface of a material (adsorbent). This phenomenon plays a pivotal role in numerous industries. For instance, it is widely applied in water treatment facilities for the purification of water, in air filters to improve air quality, and in the automotive industry within catalytic converters to reduce harmful emissions. The adsorption of compounds is usually quantified by measuring the adsorption isotherm a given adsorbate/adsorbent combination. The objective of NISTADS is two-fold: 1) to collect adsorption isotherms data from the NIST/ARPA-E Database of Novel and Emerging Adsorbent Materials (https://adsorption.nist.gov/index.php#home) through their dedicated API; 2) build a model that can accurately predict the adsorbed amount of a specific guest-host combination under various conditions, by leveraging the data from the NIST/ARPA-E Database.

This could have significant implications for industries that rely on these materials, potentially leading to more efficient processes and better materials design. As such, this project takes a different approach compared to fitting adsorption data with theoretical model for adsorption constants calculation, instead proposing the use of a deep learning approach to understand adsorption isotherm patterns by leveraging a large volume of experimental data.

![Adsorbent material](assets/5A_with_gas.png)  

### 1.2 Supplementary information
Further information are available in the `docs` folder (to be added).


## 2. Adsorption datasets
The user can either collect data regarding adsorbent materials and adsorbate species or fetch adsorption isotherm experimental data directly. Experiments are identified by name upon building the entire database experiments index from the API endpoint. Furthermore, NISTADS exploits the PUG REST API (see https://pubchempy.readthedocs.io/en/latest/ for more information) to enrich the adsorbate species dataset with molecular properties (such as molecular weight, canonical smiles, complexity, heavy atoms, etc.). Conveniently, the app will split adsorption isotherm data in two different datasets (Single Component ADSorption and Binary Mixture ADSorption). Since NISTADS is focused on predicting single component adsorption isotherms, it will make use of the single component dataset (SCADS) for the model training.

The collected data is saved locally in 4 different .csv files. Adsorption isotherm datasets and guest/host related data are saved in `resources/datasets`. Adsorption isotherms data will include the experiments datasets for both single component and binary mixture measurements.

### 2.1 Data preprocessing
The extracted data undergoes preprocessing via a tailored pipeline, which starts with filtering out undesired experiments. These include experiments featuring negative values for temperature, pressure, and uptake, or those falling outside predefined boundaries for pressure and uptake ranges (refer to configurations for specifics). Pressure and uptakes are standardized to a uniform unit—Pascal for pressure and mol/g for uptakes. Following this refinement, the physicochemical attributes of the absorbate species are unearthed through the PUBCHEM API. This enriches the input data with molecular properties such as molecular weight, the count of heavy atoms, covalent units, and H-donor/acceptor statistics. Subsequently, features pertaining to experiment conditions (e.g., temperature) and adsorbate species (physicochemical properties) are normalized. Names of adsorbents and adsorbates are encoded into integer indices for subsequent vectorization by the designated embedding head of the model. Pressure and uptake series are also normalized, utilizing upper boundaries as the normalization ceiling. Additionally, initial zero measurements in pressure and uptakes series are removed to mitigate potential bias towards zero values. Finally, all sequences are reshaped to have the same length using post-padding with a specified padding value (defaulting to -1 to avoid conflicts with actual values) and then normalized.

## 4. Installation
The installation process is designed for simplicity, using .bat scripts to automatically create a virtual environment with all necessary dependencies. Please ensure that Anaconda or Miniconda is properly installed on your system before proceeding.

- To set up the environment, run `scripts/environment_setup.bat`. This script installs Keras 3 with pytorch support as backend, and includes includes all required CUDA dependencies to enable GPU utilization (CUDA 12.1).
- **IMPORTANT:** if the path to the project folder is changed for any reason after installation, the app will cease to work. Run `scripts/package_setup.bat` or alternatively use `pip install -e . --use-pep517` from cmd when in the project folder (upon activating the conda environment).

### 3.1 Additional Package for XLA Acceleration
XLA is designed to optimize computations for speed and efficiency, particularly beneficial when working with TensorFlow and other machine learning frameworks that support XLA. Since this project uses Keras 3 with PyTorch as backend, the approach for optimizing computations for speed and efficiency has shifted from XLA to PyTorch's native acceleration tools, particularly TorchScript. This latter allows for the compilation of PyTorch models into an optimized, efficient form that enhances performance, especially when working with large-scale machine learning models or deploying models in production. TorchScript is designed to accelerate both CPU and GPU computations without requiring additional environment variables or complex setup.

For those who wish to use Tensorflow as backend in their own fork of the project, XLA acceleration can be globally enables across your system setting an environment variable named `XLA_FLAGS`. The value of this variable should be `--xla_gpu_cuda_data_dir=path\to\XLA`, where `path\to\XLA` must be replaced with the actual directory path that leads to the folder containing the nvvm subdirectory. It is crucial that this path directs to the location where the file `libdevice.10.bc` resides, as this file is essential for the optimal functioning of XLA. This setup ensures that XLA can efficiently interface with the necessary CUDA components for GPU acceleration.

## 3. How to use
Within the main project folder (NISTADS) you will find other folders, each designated to specific tasks. 

### Resources
This folder is used to organize the main data for the project, including all collected data from the NIST database (which is saved in `resources/datasets`), and the app logs that are saved in `resources/logs`.

### Database
Contains the scripts developed to extract data from the NIST DB and organise them into a readable .csv format. Data is collected through the NIST API in a concurrent fashion, allowing for fast data retrieval by selecting a maximum number of parallel HTTP requests.

- Run `retrieve_experiments_dataset.py` to fetch data for adsorption experiments
- Run `retrieve_materials_dataset.py` to respectively fetch data for adsorption experiments or for the guest/host entries. The data collection operation may take long time due to the large number of queries to perform, and it heavily depends on your internet connection performance (more than 30k experiments are available as of now). You can select a fraction of data that you wish to extract (guest, host, or experiments data), and you can also split the total number of adsorption isotherm experiments in chunks, so that each chunk will be collected and saved as file iteratively. Use the notebook `dataset_analysis.ipynb` to perform explorative data analysis on the collected datasets.

### Experimental
Contains experimental features to integrate further information into the dataset. Description of chemicals (both adsorbate species and adsorbent materials) can be generated using the pretrained GPT2 model using `gpt_enhancement.py`. Due to the model limitations, description may not be very accurate and lack context for more complex molecules and materials. 

### Inference
Here you can find the necessary files to run pretrained models in inference mode and use them to extract major features from images

- Run `images_encoding.py` to use the pretrained encoder from a model checkpoint to extract abstract representation of image features in the form of lower-dimension embeddings. 

### Training
This folder contains the necessary files for conducting model training and evaluation: 
- Run `model_training.py` to initiate the training process for the autoencoder

### Validation
Data validation and pretrained model evaluations are performed using the scripts within this folder.
- Launch the jupyter notebook `model_evaluation.ipynb` to evaluate the performance of pretrained model checkpoints using different metrics.
- Launch the jupyter notebook `data_validation.ipynb` to validate the available data with different metrics.

### 4.1 Configurations
For customization, you can modify the main configuration parameters using `settings/configurations.json`

#### General Configuration
The script is able to perform parallel data fetching through asynchronous HTML requests. However, too many calls may lead to server busy errors, especially when collecting adsorption isotherm data. Try to keep the number of parallel calls for the experiments data below 50 concurrent calls and you should not see any error!

| Setting               | Description                                           |
|-------------------------------------------------------------------------------|
| GUEST_FRACTION        | fraction of adsorbate species data to fetch           |
| HOST_FRACTION         | fraction of adsorbent materials data to fetch         |
| EXP_FRACTION          | fraction of adsorption isotherm data to fetch         |
| PARALLEL_TASKS_GH     | parallel calls to get guest/host data                 |
| PARALLEL_TASKS_EXP    | parallel calls to get experiment data                 |

                                        
## License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.


#### Dataset Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| SAMPLE_SIZE        | Number of samples to use from the dataset                |
| VALIDATION_SIZE    | Proportion of the dataset to use for validation          |
| IMG_NORMALIZE      | Whether to normalize image data                          |
| IMG_AUGMENT        | Whether to apply data augmentation to images             |
| SPLIT_SEED         | Seed for random splitting of the dataset                 |

#### Model Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| IMG_SHAPE          | Shape of the input images (height, width, channels)      |
| SAVE_MODEL_PLOT    | Whether to save a plot of the model architecture         |

#### Training Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| EPOCHS             | Number of epochs to train the model                      |
| LEARNING_RATE      | Learning rate for the optimizer                          |
| BATCH_SIZE         | Number of samples per batch                              |
| MIXED_PRECISION    | Whether to use mixed precision training                  |
| USE_TENSORBOARD    | Whether to use TensorBoard for logging                   |
| XLA_STATE          | Whether to enable XLA (Accelerated Linear Algebra)       |
| ML_DEVICE          | Device to use for training (e.g., GPU)                   |
| NUM_PROCESSORS     | Number of processors to use for data loading             |
| PLOT_EPOCH_GAP     | Epochs skipped between each point of the training plot   |

#### Evaluation Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| BATCH_SIZE         | Number of samples per batch during evaluation            |    
                    
 
## License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.




