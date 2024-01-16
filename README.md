# NISTADS-data-collection

## Project description
This is a python application that extracts adsorption isotherms data from the NIST adsorption database (https://adsorption.nist.gov/index.php#home) using their dedicated API. The application firstly extracts data regarding adsorbent materials and adsorbate species and generate two separate datasets, then collects the adsorption isotherm experimental data from the database. The dataset of adsorbate species is further modified adding molecular properties (such as molecular weight, canonical smiles, complexity, heavy atoms, etc.) through the PUG REST API (see https://pubchempy.readthedocs.io/en/latest/ for more information). The extracted data is organized in four different tables, which can be eventually saved as .csv files locally or to a S3 bucket if used with AWS services (see configurations.py):

### Extracted data summary

**Adsorbents data:** data regarding adsorbent materials 

**Adsorbates data:** data regarding adsorbate species

**Single component isotherms:** collection of adsorption isotherm experiments of single component

**Binarey mixture isotherms:** collection of adsorption isotherm experiments of binary mixture

## How to use
Run the main file NISTADS_composer.py to launch the application and show the main menu. Navigate the menu to select from the following options:

**1) Collect guest-host data:** Collect data regarding adsorbent materials and adsorbate species

**2) Collect adorption isotherm data:** Collect adsorption isotherm data from experiments database

**3) Exit and close**

The data fetching operation may take long time due to the large number of queries to perform, and it heavily depends on your internet connection performance (more than 30k experiments are available as of now). You can select a fraction of data that you wish to extract (guest, host of experiments data), therefor decreasing the estimated time till completion. Moreover, you can also split the total number of adsorption isotherm experiments in chunks, so that each chunk will be collected and saved as file prior to go for the next data subset.

### Configurations
The configurations.py file allows to change the script configuration. The following parameters are available:

- `guest_fraction:` fraction of adsorbate species data to be fetched
- `host_fraction:` fraction of adsorbent materials data to be fetched
- `experiments_fraction:` fraction of adsorption isotherm data to be fetched
- `chunk_size:` fraction of data chunks to extract and save (adsorption isotherm data)
- `output_type:` select where to save data (HOST for local files, S3 for AWS S3 buckets)
- `S3_bucket_name:` the AWS S3 bucket name
- `region_name:` the AWS region where your S3 bucket is located

### Requirements
This application has been developed and tested using the following dependencies (Python 3.10.12):

- `art==6.1`
- `pandas==2.0.3`
- `PubChemPy==1.0.4`
- `requests==2.31.0`
- `tqdm==4.66.1`
- `boto3==1.29.1`

These dependencies are specified in the provided `requirements.txt` file to ensure full compatibility with the application. 

## License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.

