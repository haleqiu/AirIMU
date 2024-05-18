# AirIMU
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-yellow.svg)](./LICENSE)
[![AirIMU website](https://airimu.github.io/)

![ALTO](./doc/alto.gif)


## Installation

This work is based on pypose. Follow the instruction and install the newest realase of pypose:
https://github.com/pypose/pypose


## Dataset

Download the Euroc dataset from:

https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

Remember to reset the `data_root` in `configs/datasets/BaselineEuroc/Euroc_1000.conf`.

## Train

Easy way to start the training using the exisiting configuration.

```
python train.py --config configs/exp/codenet.conf

```
