# Starting-Small

The goal of this repository is to demonstrate that language input ordered by age of the child improves category learning in a simple RNN language model.
It contains code to train RNNs on age-ordered and age-reversed child-directed speech.
This project exists for reference only, and is not designed to be used for answering other research questions.
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

The Wikipedia text file was created using [CreateWikiCorpus]() to extract raw text from a August 2019 Wikipedia dump file, resulting in almost 6M articles.

TODO - describe process of creating Wiki text

## Usage

The code is designed to run on multiple machines, at the UIUC Learning & Language Lab using a custom job submission system called [Ludwig](https://github.com/phueb/Ludwig).

Alternatively, the text files may be obtained from here.
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
* RNN weights are initialized with a uniform distribution, rather than a truncated normal distribution. 
The range of the uniform distribution, is equivalent to the standard deviation of the truncated normal distribution (sqrt(1/hidden_size))
* the train/test split was removed; all data is included during training
 

## Compatibility

This repository is under active development. 
Tested on Python 3.6.
