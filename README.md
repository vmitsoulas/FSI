# FEniCS Fluid-Structure Interaction (FSI) Benchmark

This repository contains a Python implementation of the Hron & Turek Fluid-Structure Interaction benchmark using the FEniCS computing platform. 

## Overview 

This project comprises an implementation of a fully-coupled two-way Fluid-Structure Interaction (FSI) numerical scheme using mixed elements to simulate the Hron & Turek problem. FEniCS, a popular computing platform for solving partial differential equations (PDE), is used to solve this benchmark problem. The code could serve as a foundation for more complex FSI problems.

## Installation

To run this project, you will need Python 3.6 or higher and FEniCS v2019.2 installed on your system.

Install latest version of FEniCS (NOT to be confused with FEniCSx):

```shell
# Ubuntu
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install --no-install-recommends fenics

# Docker
docker pull quay.io/fenicsproject/stable:current
```
## Usage

To run the FSI code:

```shell
cd FSI
python3 fsi.py
```
