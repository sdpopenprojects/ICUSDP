#ICUSDP



# Introduction

This code is an implementation of the paper, which is described in:
“Interpretable Clustering-based Unsupervised Software Defect Prediction”



# Project Structure

This directory contains the following files:
* algorihtms: Contains the core implementation of our ICUSDP framework and baseline methods.
* data: Contains the datasets used in our experiments, including 28 versions of JIRA projects with software metrics.
* test: Main scripts and entry points to run the experiments.
* utilities: a file folder contains some utility functions
* result_ICUSDP: Contains all experimental results of our proposed ICUSDP framework.
* result_baseline: Contains all experimental results of both supervised and unsupervised baseline methods.
* visual：Contains all visualization results. 
* supplementary: Contains comprehensive data tables for overall performance comparison.
 


# Usage

Please run demo_INTC.py to obtain the prediction resutls of our approaches.



# Requirements

Our framework is developed and tested under the following environment:
* OS: Windows 10
* Language: Python 3.9.25
* Key Dependencies:
  * torch (PyTorch)
  * scikit-learn
  * matplotlib



# NOTE

The software is free for academic use only, and shall not be used,
rewritten, or adapted as the basis of a commercial product without first obtaining permission from the authors.

The authors make no representations about the suitability of this software for any purpose. 
It is provided "as is" without express or implied warranty.

