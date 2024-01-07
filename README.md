# ML-Network-Intrusion-Detection

Given a packet of data detect if the packet is normal or bad using machine learning

Currently on Step 8.

Step 1: Understand the Problem
Research network traffic analysis, anomaly detection techniques, and the types of anomalies (e.g., intrusion attempts, unusual data transfers).

Step 2: Data Collection and Preparation
part A: Choose a Dataset: The KDD Cup 1999 dataset is a classic choice for intrusion detection.
Part B: Data Preprocessing: Clean and prepossess the dataset. Remove duplicates, handle missing values, and convert categorical data into numerical formats if necessary.

Step 3: Feature Extraction
Extract Relevant Features: Select features that are relevant for anomaly detection, such as source/destination IP addresses, ports, protocol type, packet size, timestamps, etc.

Step 4: Exploratory Data Analysis (EDA)
Visualize Data: Create histograms, scatter plots, and other visualizations to understand the distribution of features and identify any patterns or outliers.

Step 5: Model Selection
Choose Anomaly Detection Algorithm: Select an appropriate machine learning algorithm for anomaly detection. Common choices include Isolation Forest, One-Class SVM, and Autoencoders, Multiple classifiers. Try other choices as well, possible unsupervised algos
Train/Test Split: Split  dataset into training and testing sets. Anomalies are usually rare, so ensure a balanced representation of both normal and anomalous data in the training set.

Step 6: Model Training
Train the Anomaly Detection Model: Train the chosen anomaly detection algorithm using the training dataset. Make sure to only use the normal data for training the model.

Step 7: Model Evaluation
Test the Model: Use the testing dataset to evaluate the model's performance. The model should identify anomalies as deviations from the learned normal patterns.
Metrics: Common metrics for anomaly detection include precision, recall, F1-score, False positives, False negatives, ect.

Step 8: Visualization and Interpretation
Visualize Anomalies: Create visualizations that highlight the detected anomalies in the network traffic data.

Step 9: Deployment and Real-Time Monitoring
Integration: consider how the model can be integrated into a live network environment.
Monitoring: Develop a system that can continuously monitor incoming network traffic and classify it as normal or anomalous.
