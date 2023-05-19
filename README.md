# Unsupervised Flow Refinement near Motion Boundaries (BMVC2022)
This repository contains the implementation of our paper titled [*Unsupervised Flow Refinement near Motion Boundaries*](https://arxiv.org/abs/2208.02305), which has been accepted by BMVC-2022.

## Requirements
This code has been developed under Python 3.10, PyTorch 1.12.0, and CUDA 11.3 on Ubuntu 18.04. The enviroment can be built as follows using conda and pip:
'''shell
conda create -n mbflow python=3.10
conda activate mbflow
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
'''