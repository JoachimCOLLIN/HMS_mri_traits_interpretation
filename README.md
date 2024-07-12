# Polygenic Risk Score Prediction from Organ MRI Images
During my internship, I worked on a project that built upon the research of Alan Le Goallec (https://github.com/JoachimCOLLIN/Main-pipeline-and-Images-based-models-pipeline.git), a former student at Ecole des Ponts ParisTech (2011-2015) and a doctoral candidate at Harvard Medical School (2015-2021). His thesis focused on predicting a patient’s age using MRI scans, time series data, and numerical features through Deep Learning methods. Alan developed a complex ten-step pipeline, testing and comparing numerous Convolutional Neural Networks (CNNs). The goal was to compare predicted organ ages with actual age to assess a patient’s health status. The entire project was implemented in Python (using TensorFlow 2 and Scikit-Learn) and Bash.

During my internship in Chirag Patel's laboratory in the Biomedical Informatics Department of the Harvard Medical School, I aimed to adapt and enhance the codebase developed by Alan Le Goallec. Instead of predicting patient age, my focus shifted to using patient organ MRI scans to predict a Polygenic Risk Score (PRS). PRS represents genetic predisposition to specific traits and diseases.
I explored different state-of-the-art CNN models to understand which features influenced predictions. For instance, I investigated predicting an individual’s genetic risk for cardiovascular accidents using PRS. If successful, I aimed to identify relevant image regions according to the model’s predictions, potentially uncovering valuable insights.
Combining Multiple PRS: I also considered complex, multifunctional structures to combine organ-specific PRS. For example, analyzing cardiovascular genetic risks based on combined PRS from various organs.
Implementation Details
Programming Languages: Python (TensorFlow 2 and Scikit-Learn) and Bash.
Pipeline: Adapted from Alan’s work, with modifications for PRS prediction.
MRI Data: Used organ-specific MRI scans for PRS estimation.
