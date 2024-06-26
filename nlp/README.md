# Setup Steps
## 1. Set up GCP environment
- Under the `Compute Engine`, create an VM instance. 
- Set Machine type to be `n1-standard-8`, GPU type to be `NVIDIA Tesla P100`
- Set boot disk to Operating System `Deep Learning on Linux` and Version `Deep Learning Image: PyTorch 1.4.0 and fastai m58`. 
## 2. Setup conda environment
- Set up Python environment by running `conda env create -f bva-final-env.yaml`. 
- Activate the environment with `conda activate bva-final`.
## 3. What are the files
- `dataset_vocab.py`: Largely modified from Matthias's code. Modified to add attention and changed return type to integrate with training code.
- `dataset_build.py`: Largely modified from Matthias's code. Added the function for caching citation indices.
- `preprocessing.py `: Largely modified from Matthias's code. Added `MetadataProcessor` to preprocess the metadata file.
- `util.py`: This includes a set of utility functions.
- `bilstm.py`: Defines the BiLSTM model and its code for evaluation and testing.
- `rooberta.py`: Defines the Roberta model and its code for evaluation and testing.
- `train.py`: This is the main entry point. It sets up the training based on the parameters passed in the configuration file. It can also load a trained model and run the test.
- `bva-final-env.yaml`: File to setup conda environment.
- `config-no-meta.yaml`: Example file for the training and testing configuration.
## 3. Get the preprocessed data and metadata.
- Get caselaw access metadata `f3d.jsonl` under Google Drive.
- Get metadata file `appeals_meta_wscraped.csv` from Google Drive.
- [Optional] Get full dataset. Not required if preprocessed cases already exist.
- Get the preprocessed cases.
- Get the text file with training/evaluation ids.
- Get the text file with partitioned test ids.
- Get the built vocabulary and reduced vocabulary. 
## 4. Set up parameters.
Set up the paths and parameters in the config file. An example is the `config-no-meta.yaml`.
## 5. Run the code for training/testing
 ```sh
 $ python train.py --config config-no-meta.yaml
 ```
