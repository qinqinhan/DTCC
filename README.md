# Deep Time Series Contrastive Clustering with Cross-view Reliable Cluster Diffusion
The source code is for reproducing experiments of the paper entitled "Deep Time Series Contrastive Clustering with Cross-view Reliable Cluster Diffusion"
# Datasets
The UCR dataset used in the paper are available at : http://www.timeseriesclassification.com/ .
In order to read the data intuitively and save space, we converted the data into csv format and compressed it. The processed data is available at: (https://drive.google.com/file/d/1oojgNxyefGhSK9ZlN-Vb-xm3DKpKALCl/view?usp=drive_link)

# Usage

## Install packages

You can use your favorite package manager, or create a new environment of python 3.6 or greater and use the packages listed in requirements.txt
`pip install -r requirements.txt`

## Setting parameters

Set parameters in file config/DTCC.yaml. 
Hyperparameters were adjusted using grid search and optimized using the Adam optimizer.
## Run

`python main.py -f config/DTCC.yaml`
