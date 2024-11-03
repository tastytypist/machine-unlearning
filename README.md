# On Performance Comparison Between Strong Machine Unlearning Algorithms for Logistic Regression Credit Assessment Models
Proceedings of 2024 IEEE International Conference on Data and Software Engineering (ICoDSE)

## Abstract
The enactment of the Personal Data Protection (PDP) regulation in Indonesia 
requires financial service institutions to erase debtors’ personal data upon 
request. However, this is challenging to achieve when such data is implicitly 
stored in a trained machine learning model. In order to address this issue, 
machine unlearning methods have been developed to erase the influence of training 
data on model weights. In this research, the performance of two strong machine 
unlearning algorithm implementations, ε-δ Certified Removal (CR) and Projective 
Residual Update (PRU), is compared on the logistic regression credit risk 
assessment models developed in this research. Machine unlearning was performed to 
delete 10% of the model’s training data. Our results showed that ε-δ Certified 
Removal yielded a model with lower L2-distance, higher accuracy, and faster 
unlearning time compared to Projective Residual Update when machine unlearning 
was performed to delete less than 2% of the model’s training data. Conversely, 
the opposite was observed when machine unlearning was performed to delete more or 
equal to 2% of the model’s training data. Further research is required to explore 
the effect of larger training data sets with greater dimensionality on the 
performance of both algorithms.

## Keywords
machine unlearning, ε-δ certified removal, projective residual update, 
performance, credit risk assessment

## Setup
Assuming you have installed the latest version of Python,
1. ensure `pip` is installed by running `python -m ensurepip --upgrade`, then
2. install the library dependencies by running `pip install -r requirements.txt`.

## Experiment
Modify the experiment parameters in the source code as needed!

### Perform the accuracy and L2-distance experiment
```bash
python main.py
```

### Perform the runtime experiment
```bash
python runtime.py
```
