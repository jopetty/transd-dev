# Transductions

A seq2seq experimentation framework for recurrent and transformer networks.

## Installation and Usage

The transductions framework can be run locally, but it includes support for containerization via Docker. I recommend using Visual Studio Code with the Remote extension to automatically handle docker integration. If you run the framework locally, you will need to create a Conda environment for the project using the provided Condafile.
```
conda env create --file .devcontainer/environment.yaml && conda activate ml-exp
```
This will install all the requisite Python dependences via Conda and Pip.

If you run the framework in Docker, the provided Dockerfile will handle the creation of the Docker container and the installation of the Conda environment.

### Configuration

Transductions uses the Hydra framework for experiment configuration. All configuration is managed by YAML files in the `conf/` directory. Hydra composes these configuration files into a single experiment configuration block, and you can specify specific overrides using the built-in CLI (see hydra.cc for examples).

Here, all configuration should be managed by an `experiment` config file, which lives in `conf/experiment`. This will specify a number of things for the training script, including:
- dataset
- model type
- callbacks (a PyTorch Lightning feature)
- training hyperparameters

### Training

To train using a given experimental paradigm, run
```
python run.py experiment=FILE
```
where `conf/experiment/FILE.yaml` is a configuration file specifying the experiment parameters.

### Integrations

Transductions will, by default, log training runs using [Weights and Biases](wandb.ai). You will be prompted to enter an authentication token on the first run. To remove this functionality, override the `logger` value to `null` in your experiment configuration file.

To sync local runs to your W&B account, copy your [authentication token](https://wandb.ai/authorize) into a root-level file called `.env` containing the key as the value for the `WANDB_API_KEY` environment variable:
```
WANDB_API_KEY=##################################
```
Note that this file is explicitly excluded from git tracking since it contains private key information. Never track this file in git.
