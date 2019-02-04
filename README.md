# PROPS Model 
 

The **PROPS model** (or probabilistically personalized black-box sequence model) is a transfer learning mechanism for modeling sequential data. It takes the feedforward predictions of a pre-trained and black-box sequence model (e.g. an RNN) and probabilistically perturbs these predictions to fit a new situation.  In this way, the PROPS model customizes the baseline sequence model into a personalized sequence model.  This customization happens in a streaming/online manner.  For more information, see the [paper](paper/transfer_learning.pdf).


## Setup 

### Setup for Development

For local development, clone the repo and run commands in [Makefile](Makefile) to setup a virtualenv in `env/`:

```bash 
$ make clean-env env
```

This command will perform a psuedo-installation to `env/`.  For more information, see the Makefile.

### Dependencies
See [requirements.txt](requirements.txt).  The `make all` command will obtain dependencies (with correct versions) automatically. 

## Experiments 
The [experiments/](experiments/) directory contains code for running the experiments reported in the paper. 

Experiments can be reconstructed from the command line via

```bash 
$ make experiment1
```

and

```bash 
$ make experiment2
```

It takes a modern MacBook Pro hours to run the experiments for `make experiment1`, and seconds or minutes for `make experiment2`. Alternatively, you can run both the long-running and short-running experiments by executing
```bash
$ make experiment
```

These commands will write plots to directory `plots/`.


## Tests
Unit tests are provided in the [tests/](tests/) directory, and can be run from the command line via 

```bash 
$ make test
```
There are also tests for the streaming HMM model.
