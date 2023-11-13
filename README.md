# Cleanba: A Reproducible and Efficient Distributed Reinforcement Learning Platform


Cleanba is **Clean**RL-style implementation of DeepMind's Sebul**ba** distributed training platform, but with a few different design choices to make distributed RL more reproducible and transparent to use.


## Get started

Prerequisites:
* Python >=3.8
* [Poetry 1.3.2+](https://python-poetry.org)
* CUDA 11.2+
* CuDNN 8.2+


### Installation:
```
poetry install
poetry run pip install --upgrade "jax[cuda11_cudnn82]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
poetry run python cleanba/cleanba_ppo.py --help
poetry run python cleanba/cleanba_ppo.py
poetry run python cleanba/cleanba_impala.py --help
poetry run python cleanba/cleanba_impala.py
```