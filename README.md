# Multi-Objective Maximum a posteriori Policy Optimization (MO-MPO)
## Introduction
This repository implements **MO-MPO** algorithm, introduced in ([Abdolmaleki, Huang et al., 2020](https://arxiv.org/abs/2005.07513)), by using PyTorch.

## Getting Started
- First, download this repo to your local machine,
    ```bash
    git clone https://github.com/ToooooooT/MOMPO.git
    ```
- Next, change to the repo directory, and you need to install all the required Python packages,  
  (In case you would not like to build a local virtual environment, you can just skip the second line.)
    ```bash
    cd MOMPO/
    python -m venv .venv
    pip install -r requirements.txt
    ```
- To run **MO-MPO** in the **"Deep Sea Treasure"** environment, 
    ```bash
    python run_cat_mompo.py
    ```
- To run **MO-MPO** in the **"Humanoid-run"** environment,
    ```bash
    python run_cont_mompo.py
    ```