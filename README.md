# NISTCOLLECT: NIST/ARPA-E dataset composer

## 1. Project Overview
NISTADS is a python application developed to extract adsorption isotherms data from the NIST/ARPA-E Database of Novel and Emerging Adsorbent Materials (https://adsorption.nist.gov/index.php#home) through their dedicated API. The user can either collect data regarding adsorbent materials and adsorbate species or fetch adsorption isotherm experimental data. Experiments are identified by name upon building the entire database experiments index from the API endpoint. Furthermore, NISTADS exploits the PUG REST API (see https://pubchempy.readthedocs.io/en/latest/ for more information) to enrich the adsorbate species dataset with molecular properties (such as molecular weight, canonical smiles, complexity, heavy atoms, etc.). Eventually, the adsorption isotherm dataset is split into two datasets, one containing data on single component adsorption and the other including experiments with binary mixtures.

![Adsorbent material](assets/5A_with_gas.png)  

## 2. Adsorption datasets
The collected data is saved locally in 4 different .csv files, located in the `NISTADS/data` folder. Adsorption isotherm datasets are saved in `NISTADS/data/experiments`, while the adsorbents and adsorbates datasets are saved into `NISTADS/data/materials`. The former will include the experiments datasets for both single component and binary mixture measurements, while the latter will host datasets on guest and host species. 

## 2. Installation 
The installation process is designed for simplicity, using .bat scripts to automatically create a virtual environment with all necessary dependencies. Please ensure that Anaconda or Miniconda is installed on your system before proceeding.

- The `scripts/environment_setup.bat` file, located in the scripts folder, offers a convenient one-click solution to set up your virtual environment.
- **IMPORTANT:** run `scripts/package_setup.bat` if the path to the project folder is changed for any reason after installation, or the app won't work!

## 3. How to use
Within the main project folder (NISTCOLLECT) you will find other folders, each designated to specific tasks. 

### Datasets
Contains the scripts developed to extract data from the NIST DB and organise them into a readable .csv format. Data is collected through the NIST API in a concurrent fashion, allowing for fast data retrieval by selecting a maximum number of parallel HTTP requests.

- Run `compose_experiments_dataset.py` to fetch data for adsorption experiments
- Run `compose_materials_dataset.py` to respectively fetch data for adsorption experiments or for the guest/host entries. The data collection operation may take long time due to the large number of queries to perform, and it heavily depends on your internet connection performance (more than 30k experiments are available as of now). You can select a fraction of data that you wish to extract (guest, host, or experiments data), and you can also split the total number of adsorption isotherm experiments in chunks, so that each chunk will be collected and saved as file iteratively. Use the notebook `NISTADS/data/dataset_info.ipynb` to perform explorative data analysis on the collected datasets.

### Resources
This folder is used to organize data and results for various stages of the project, including data validation, model training, and evaluation. Here are the key subfolders:

**experimental:** contains experimental features to integrate further information into the dataset. Description of chemicals (both adsorbate species and adsorbent materials) can be generated using the pretrained GPT2 model using `NISTADS/experimental/gpt_enhancement.py`. Due to the model limitations, description may not be very accurate and lack context for more complex molecules and materials. 

### 3.1 Configurations
For customization, you can modify the main script parameters via the `NISTADS/commons/configurations.py` file. 

| Category              | Setting               | Description                                           |
|-----------------------|-----------------------|-------------------------------------------------------|
| **Data settings**     | GUEST_FRACTION        | fraction of adsorbate species data to be fetched      |
|                       | HOST_FRACTION         | fraction of adsorbent materials data to be fetched    |
|                       | EXP_FRACTION          | fraction of adsorption isotherm data to be fetched    |
|                       | CHUNK_SIZE            | fraction of data chunks to extract and save           |
| **Series settings**   | MIN_POINTS            | Minimum number of measurements per experiment         |
|                       | MAX_PRESSURE          | Max pressure to consider (in Pa)                      |
|                       | MAX_UPTAKE            | Max uptake to consider (in mol/g)                     |
                                         
## License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.

