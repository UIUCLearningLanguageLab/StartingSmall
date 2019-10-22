# Starting-Small

Research code to train RNNs on age-ordered child-directed speech. 
The main research question is to
1) test whether language input ordered by age of the child improves category learning, and
2) explain any learning benefit in terms of RNN learning dynamics

## Documentation

Theoretical motivation and detailed analyses of the results can be found in my master's thesis, submitted in August 2018.
The thesis will become available on my [personal website](http://philhhuebner.com) in early 2020.

## History

Initial work began in 2016, at the University of California, Riverside, under the supervision of Prof Jon Willits.

In an effort to simplify the code used in this repository, a major rewrite was undertaken in October 2019.
The code was ported from tensorflow 1.12 to pytorch.
One consequence of this change was the custom RNN architecture was replaced by a more standard architecture.
Specifically, prior to October 2019, embeddings were directly added to the hidden layer.
In the standard RNN architecture, embeddings undergo an additional transformation step before being added to the hidden layer.

 

## Usage

The code is designed to run on multiple machines, at the UIUC Learning & Language Lab using a custom job submission system called [Ludwig](https://github.com/phueb/Ludwig).

Alternatively, the text files may be obtained from here.
To run the default configuration, call `starting_small.job.main` like so:

```python
from startingsmall.job import main
from startingsmall.params import param2default

main(param2default)  # runs the experiment in default configuration
```

## Compatibility

This repository is under active development. 
Tested on Python 3.6.
