This directory holds python scripts that were used to develop the LGFE-GCN physics/ML hybrid model as described in the paper "Ligand-Protein Specificity Predictions Using Physics-based Ligand Docking Scoring Features in a Graph Neural Network":
"lgfe-gcn-hyper-tunning.py" was used for hyperparameter tunning to get a optimal set of hyperparameters
With the optimal set of hyperparameters, "lgfe-gcn-train.py" was used to train the model on the training set and save the model to "model.pt"
"test/lgfe-gcn-test.py" was used to run the test set using the trained model "model.pt" and write predictions to "pred.csv"
"sdf/" and "test/sdf/" hold preprocessed training and test set data and experimental binding data "sdf/exp.csv" and "test/sdf/test.csv"
