# Source Code for Manuscript

This repository contains the source code, simulation scripts, and trained models used in the manuscript submitted for review. The code validates the performance of the **Physics-Based Transformer Model (PITM)** designed for battery aging estimation.

## Repository Structure

The repository is organized into three main directories, corresponding to the methodology and validation steps described in the paper:

### 1. General new approach
This folder contains the core implementation of the generalized **PITM_G** model.
* **Contents:**
    * Jupyter Notebook scripts for the generalized approach.
    * Subfolders containing the corresponding trained models.
    * Data files required for training and validation.

### 2. Single group
This folder contains the baseline or specific group studies.
* **Structure:** It contains **5 subfolders**.
* **Contents:**
    * Each subfolder contains the scripts for single PITM models.
    * Corresponding trained models for each specific group.

## Requirements
To run these simulations, the following are required:
* **Python 3.x**
* **Jupyter Notebook**
* **Key Libraries:** NumPy, SciPy (for .mat files), Matplotlib, and PyTorch/TensorFlow (depending on the transformer implementation).

## Usage
1.  Clone this repository.
2.  Navigate to the specific folder of interest (e.g., `General new approach` for the main model).
3.  Open the `.ipynb` file in Jupyter Notebook.
4.  Run the cells sequentially to reproduce the results presented in the manuscript.

## Contact
For questions regarding the code or data, please contact the corresponding author via the email provided in the manuscript.