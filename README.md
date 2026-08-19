# LGFE-GCN

#### Python scripts that were used to develop the LGFE-GCN physics/ML hybrid model as described in the paper:
#### "Ligand-Protein Specificity Predictions Using Physics-based Ligand Docking Scoring Features in a Graph Neural Network"

## Contents:
* lgfe-gcn-hyper-tunning.py: hyperparameter tunning script
* lgfe-gcn-train.py: model training script using the optimal set of hyperparameters determined using the previous script
* test/lgfe-gcn-test.py: run the test set predictions
* model.pt: trained GCN model
* sdf/*.h5 : preprocessed training set
* sdf/exp.csv: experimental binding data for the training set
* test/sdf/*.h5 : preprocessed test set
* test/sdf/test.csv : experimental binding data for the test set
* environment.yml : required Python packages and dependencies


## Example:

```
cd test
python3 lgfe-gcn-test.py
```
This will run the test set prediction and will print out the following metrics:
```
ptp1b: RMSE = 1.475, R = 0.195  (n=23)
syk: RMSE = 1.705, R = -0.134  (n=44)
bclxl: RMSE = 3.053, R = 0.434  (n=19)
Average RMSE across groups: 2.078
Average Pearson R across groups: 0.165
```
