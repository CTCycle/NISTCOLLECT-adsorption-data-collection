# NISTADS: NIST/ARPA-E dataset composer

## 1. Project Overview
This is a python application that extracts adsorption isotherms data from the NIST adsorption database (https://adsorption.nist.gov/index.php#home) using their dedicated API. The application firstly extracts data regarding adsorbent materials and adsorbate species and generate two separate datasets, then collects the adsorption isotherm experimental data from the database. The dataset of adsorbate species is further modified adding molecular properties (such as molecular weight, canonical smiles, complexity, heavy atoms, etc.) through the PUG REST API (see https://pubchempy.readthedocs.io/en/latest/ for more information). The extracted data is organized in four different tables, which can be eventually saved as .csv files locally.

The returned files will be saved in their respective folders in `data\experiments` and `data\materials`. The former will include the experiments datasets for both single component and binary mixture measurements, while the latter will host datasets on guest and host species. 

## 2. Installation 
First, ensure that you have Python 3.10.12 installed on your system. Then, you can easily install the required Python packages using the provided requirements.txt file:

`pip install -r requirements.txt` 

## 3. How to use
The project is organized into subfolders, each dedicated to specific tasks. The `utils/` folder houses crucial components utilized by various scripts. It's critical to avoid modifying these files, as doing so could compromise the overall integrity and functionality of the program.

**Data:** run `compose_experiments_dataset.py` or `compose_materials_dataset.py` to respectively fetch data for adsorption experiments or for the guest/host entries. The data collection operation may take long time due to the large number of queries to perform, and it heavily depends on your internet connection performance (more than 30k experiments are available as of now). You can select a fraction of data that you wish to extract (guest, host, or experiments data), and you can also split the total number of adsorption isotherm experiments in chunks, so that each chunk will be collected and saved as file iteratively.

**Preprocessing:** run `preprocess_dataset.py` to perform preprocssing duties on the experiments dataset, specifically that including single component adsorption isotherm data. 

### 3.1 Configurations
The configurations.py file allows to change the script configuration. 

| Category              | Setting               | Description                                                              |
|-----------------------|-----------------------|--------------------------------------------------------------------------|
| **Data settings**     | guest_fraction        | fraction of adsorbate species data to be fetched                         |
|                       | host_fraction         | fraction of adsorbent materials data to be fetched                       |
|                       | experiments_fraction  | fraction of adsorption isotherm data to be fetched                       |
|                       | chunk_size            | fraction of data chunks to extract and save                              |
| **Series settings**   | min_points            | Minimum number of measurements per experiment                            |
|                       | max_pressure          | Max pressure to consider (in Pa)                                         |
|                       | max_uptake            | Max uptake to consider (in mol/g)                                        |
                                         

## License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.

