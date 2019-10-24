# Starting-Small

The goal of this repository is to demonstrate that language input ordered by age of the child improves category learning in a simple RNN trained to predict child-directed speech.
It contains code to train RNNs on age-ordered and age-reversed child-directed speech.
This project exists for reference and replication only, and is not designed to be used for further development.
To that end, most of the complexity of the original project have been stripped.  

## Motivation & Results

Theoretical motivation and detailed analyses of the results can be found in my master's thesis, submitted in August 2018.
The thesis will become available on my [personal website](http://philhhuebner.com) in early 2020.

## Training Data

The text files used as input to the RNN have undergone a number of processing steps.
The CHILDES text file was created using [CHILDESHub](https://github.com/phueb/CHILDESHub), which performs:

1) tokenization using the default tokenizer in `spacy`
2) lowercasing
3) ordering of transcripts by the age of the target child

## Dependencies

To install all the dependencies, execute the following in a virtual environment: 

```bash
pip install -r requirements.txt
```

### Preppy v1.0.0

The text files are prepared for training using a custom Python package `Preppy`.
It is available [here](https://github.com/phueb/Preppy).
It performs no reordering of the input, and assumes instead that the lines in the text file are already in the order that they should be presented to the model.

### CategoryEval v1.0.0

Evaluation of semantic category knowledge requires the custom Python package `CategoryEval`.
It is available [here](https://github.com/phueb/CategoryEval).
It computes how well the model's learned representations recapitulate some human-created gold category structure.
By default, it returns the balanced accuracy, but F1 and Cohen's Kappa can be computed also.

### Ludwig 1.2.1 (Optional)

If you are a member of the UIUC Learning & Language lab, you can run the jobs in parallel on multiple machines.
This is recommended if multiple replications need to be run, or if no access to GPUs is otherwise available.

## Usage

The code is designed to run on multiple machines, at the UIUC Learning & Language Lab using a custom job submission system called [Ludwig](https://github.com/phueb/Ludwig).
If you have access to the lab's file server, you can submit jobs with `Ludwig`:

```bash
ludwig -c PATH_TO_PREPPY PATH_TO_CATEGORYEVAL
```

Alternatively, the corpus file has been included for users without access to the server.
Make sure to edit the path to the folder in which the text file is located by modifying `config.RemoteDirs.data`
To run the default configuration, call `starting_small.job.main` like so:

```python
from startingsmall.job import main
from startingsmall.params import param2default

main(param2default)  # runs the experiment in default configuration
```

## History

### 2016-2018
Initial work began in 2016, at the University of California, Riverside, under the supervision of Prof Jon Willits.

### 2019
In an effort to simplify the code used in this repository, a major rewrite was undertaken in October 2019.
The code was ported from tensorflow 1.12 to pytorch.
The following changes resulted due to the porting:
* the custom RNN architecture was replaced by a more standard architecture. 
Specifically, prior to October 2019, embeddings were directly added to the hidden layer.
In the standard RNN architecture, embeddings undergo an additional transformation step before being added to the hidden layer.

## Compatibility

Developed on Ubuntu 16.04 with Python 3.6
