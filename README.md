The goal of this project is to implement an email spam filter that determines whether an email is spam ("spam") or not ("ham"). This project leverages the Naive Bayes algorithm, a widely used technique for text classification, to develop an ML model that can accurately identify spam emails and automate the filtering process.

🚀 Project Workflow
Data Collection & Preprocessing

We will use a dataset of labeled spam and non-spam emails.
The dataset will be stored in Google Cloud Storage (GCS) for easy access.
Emails will be cleaned, tokenized, and transformed into numerical features for training.
Model Training & Evaluation

We will train a Naive Bayes classifier for spam detection.
Model performance will be evaluated using metrics such as accuracy, precision, recall, and F1-score.
Hyperparameter tuning will be applied to optimize performance.
MLOps Implementation

Version control for datasets and models using DVC (Data Version Control) and GitHub.
Automation of workflows using Google Cloud AI Pipelines.
Model monitoring and re-training for continuous improvement.
Model Deployment

The trained model will be deployed on Google Cloud AI Platform.
The system will expose an API endpoint for real-time email classification.