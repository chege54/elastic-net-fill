# Descriptor-guided Selection of Extracellular Vesicle Loading Strategies

This repository contains the source code and datasets supporting the manuscript: **"Descriptor-guided selection of extracellular vesicle loading strategies for small-molecule drug delivery: a mechanistically interpretable decision-support framework."**

The provided code implements a machine learning framework using Elastic Net regularization and Leave-One-Out Cross-Validation (LOOCV) to predict the loading efficiency of small-molecule drugs into extracellular vesicles (EVs). The model uses physicochemical drug descriptors to guide the selection among five common EV loading strategies.

## Overview

The core script, `std_elastic_net.py`, evaluates seven molecular features (LogP, Molecular Weight, Solubility, Hydrogen Bond Donors, Hydrogen Bond Acceptors, Polar Surface Area, and Charge) to estimate loading outcomes for the following methods:
* Passive Incubation
* Electroporation
* Saponin-assisted loading
* Freeze-Thaw cycling
* Sonication

## Repository Structure

* `std_elastic_net.py`: The main Python script containing the Elastic Net LOOCV model, feature standardization, and evaluation logic.
* `input_data/table.csv`: The primary dataset containing the molecular descriptors and experimentally measured loading efficiencies for various drugs.
* `input_data/table-filtered.csv`: An additional filtered dataset for subset analysis.
* `results.txt`: The expected standard output demonstrating the best hyperparameters, calculated coefficients, and side-by-side comparisons of measured versus estimated fill methods.
* `.vscode/launch.json`: Configuration file for debugging the script in Visual Studio Code.

## Prerequisites and Dependencies

The script requires Python 3.x and the following standard data science libraries:
* `pandas`
* `numpy`
* `scikit-learn`

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn
```

## How to Run

* Ensure that the dataset `table.csv` is located in the `input_data/` directory relative to the script.
* Execute the script from your terminal or command prompt:

```bash
python std_elastic_net.py
```

## Expected Output

When the script is executed, it will output the following to the console:

* Best Hyper-parameters: The optimal alpha and L1 ratio found for each loading method using LOOCV.
* Best Coefficients: The calculated weights for each physicochemical descriptor (plus the intercept) for all five loading methods, allowing for mechanistic interpretation of the model.
* Measured vs. Estimated values: A tabular comparison of the empirical loading efficiencies and the model's predicted values for each drug across every loading method.