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

## 3. Installation
The installation process on Windows has been designed for simplicity and ease of use. To begin, simply run `NISTADS_AutoEncoder.bat`. On its first execution, the installation procedure will automatically start with minimal user input required. The script will check if either Anaconda or Miniconda is installed on your system. If neither is found, you will need to install it manually. You can download and install Miniconda by following the instructions here: (https://docs.anaconda.com/miniconda/).

After setting up Anaconda/Miniconda, the installation script will install all the necessary Python dependencies. This includes Keras 3 (with PyTorch support as the backend) and the required CUDA dependencies (CUDA 12.1) to enable GPU acceleration. If you'd prefer to handle the installation process separately, you can run the standalone installer by executing `setup/NISTADS_installer.bat`. You can also use a custom python environment by modifying `settings/launcher_configurations.ini` and setting use_custom_environment as true, while specifying the name of your custom environment.

**Important:** After installation, if the project folder is moved or its path is changed, the application will no longer function correctly. To fix this, you can either:

- Open the main menu, select "NISTADS setup," and choose "Install project packages"
- Manually run the following commands in the terminal, ensuring the project folder is set as the current working directory (CWD):

    `conda activate NISTADS`

    `pip install -e . --use-pep517` 

### 3.1 Additional Package for XLA Acceleration
XLA is designed to optimize computations for speed and efficiency, particularly beneficial when working with TensorFlow and other machine learning frameworks that support XLA. Since this project uses Keras 3 with PyTorch as backend, the approach for optimizing computations for speed and efficiency has shifted from XLA to PyTorch's native acceleration tools, particularly TorchScript (currently not implemented). For those who wish to use Tensorflow as backend, XLA acceleration can be globally enabled setting the `XLA_FLAGS` environmental variabile with the following value: `--xla_gpu_cuda_data_dir=path\to\XLA`, where `path\to\XLA` is the actual directory path to the folder containing the nvvm subdirectory (where the file `libdevice.10.bc` resides).

## 4. How to use
On Windows, run `NISTADS.bat` to launch the main navigation menu and browse through the various options. Alternatively, you can run each file separately using `python path/filename.py` or `jupyter path/notebook.ipynb`. 

### 4.1 Navigation menu

**1) Data analysis:** run `validation/data_validation.ipynb` to perform data validation using a series of metrics for the analysis of the dataset. This feature cannot be directly started from the launcher due to unpredictable behavior of .ipynb files when executed from batch scripts.

**2) Collect adsorption data:** extract data from the NIST DB and organise them into a readable .csv format. Data is collected through the NIST API in a concurrent fashion, allowing for fast data retrieval by selecting a maximum number of parallel HTTP requests. Alternatively, one can run `database/retrieve_adsorption_data.py`

**2) Data preprocessing:** prepare the adsorption dataset for machine learning. This is done by running `preprocessing/data_preprocessing.py`

**3) Model training and evaluation:** open the machine learning menu to explore various options for model training and validation. Once the menu is open, you will see different options:
- **train from scratch:** runs `training/model_training.py` to start training an instance of the NISTADS model from scratch using the available data and parameters. 
- **train from checkpoint:** runs `training/train_from_checkpoint.py` to start training a pretrained NISTADS checkpoint for an additional amount of epochs, using pretrained model settings and data.  
- **model evaluation:** run `validation/model_validation.ipynb` to evaluate the performance of pretrained model checkpoints using different metrics. This feature cannot be directly started from the launcher due to unpredictable behavior of .ipynb files when executed from batch scripts.

**4) Predict adsorption of compounds:** Run `inference/adsorption_prediction.py` to use the pretrained NISTADS model and predict adsorption of compounds based on pressure.  

**5) NISTADS setup:** allows running some options command such as **install project packages** to run the developer model project installation, and **remove logs** to remove all logs saved in `resources/logs`. 

**6) Exit and close:** exit the program immediately

### 4.2 Resources
This folder is used to organize data and results for various stages of the project, including data validation, model training, and evaluation. Here are the key subfolders:

**datasets:** ....

**predictions:** ...

**checkpoints:** pretrained model checkpoints are stored here, and can be used either for resuming training or performing inference with an already trained model.

### Experimental
Contains experimental features to integrate further information into the dataset. Description of chemicals (both adsorbate species and adsorbent materials) can be generated using the pretrained GPT2 model using `gpt_enhancement.py`. Due to the model limitations, description may not be very accurate and lack context for more complex molecules and materials. 


## 5. Configurations
For customization, you can modify the main configuration parameters using `settings/app_configurations.json` 

#### General Configuration
The script is able to perform parallel data fetching through asynchronous HTML requests. However, too many calls may lead to server busy errors, especially when collecting adsorption isotherm data. Try to keep the number of parallel calls for the experiments data below 50 concurrent calls and you should not see any error!

| Setting               | Description                                           |
|-------------------------------------------------------------------------------|
| GUEST_FRACTION        | fraction of adsorbate species data to fetch           |
| HOST_FRACTION         | fraction of adsorbent materials data to fetch         |
| EXP_FRACTION          | fraction of adsorption isotherm data to fetch         |
| PARALLEL_TASKS_GH     | parallel calls to get guest/host data                 |
| PARALLEL_TASKS_EXP    | parallel calls to get experiment data                 |

                                        
#### Dataset Configuration (to be changed)

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




