# UCI_Heart_disease_UCI_R_machinelearning
Heart disease binary classification from clinical data (Heart disease UCI) with R

The dataset can be found at:
https://www.kaggle.com/ronitf/heart-disease-uci
Data Set Information (directly from Kaggle):
This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In
particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "goal"
field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4.
Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence
(values 1,2,3,4) from absence (value 0).
The names and social security numbers of the patients were recently removed from the database, replaced with
dummy values.
One file has been "processed", that one containing the Cleveland database. All four unprocessed files also exist
in this directory.
You have a nice dataset with a total of 303 observations, which is not a lot to build a model, in this case
hold-out validation will not be possible, so a cross-validation strategy is suggested, both for internal,
and external, validation.
Follow the following steps
1) LOOK AT THE DATA, Check your data visually, can you understand it? Can you easily spot valuable or
irrelevant information? Is it easy from a human point of view? A first assessment of the data can give
you a first intuition of wethere your problem is linearly separable or not. Generate visualizations and
comment any interesting features you encounter.
2) Think about your data splits and cross-validation strategy. How will the splits be generated? How shall
the final accuracy be computed? Basically, present your method in detail.
3) Generate PCA and LDA visualizations of your data, comment on the results.
4) Compute classifier models for a kNN, naïve Bayes and SVM classifier (you may attempt different
kernels). You may use dimensionally reduced spaces, or the full 14 variable space. Pay attention to the
internal validation requirements for every classifier (number of parameters needed to tune) and discuss
their advantages / disadvantages in relation to the dataset under study.
5) Report the accuracies of the models, you can use simple test accuracies to summarize the models, but
use ROC curves for the most interesting ones. Can you give insight about the misclassified samples? Can
you elaborate some conclusions about the problem? What difference does it make to operate in the
dimensionally reduced space, or the “full” 14 feature space?
