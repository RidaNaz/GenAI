# Machine Learning (ML)
- Subset of Artificial Intelligence (AI)
- Training algorithms learn from data to make predictions/decisions.
- **Example:** Spam detection, recommendation systems, Predictive modeling, Natural Language Processing (NLP)

## ML Branches / Types

#### 1. Supervised Learning
- Uses labeled data (input-output pairs)
- Learn a mapping between input data and output labels
- **Examples:**
    - Image classification (e.g., cat vs. dog)
    - Sentiment analysis (e.g., positive vs. negative review)
    - Regression tasks (e.g., predicting house prices)

#### 2. Unsupervised Learning
- Works with unlabeled data, finds hidden patterns
- Discover patterns, relationships, or structure in data
- **Examples:**
    - Clustering customers by behavior or demographics
    - Dimensionality reduction (e.g., PCA, t-SNE)
    - Anomaly detection (e.g., fraud detection)

#### 3. Reinforcement Learning
- Use trial and error to learn from interactions with an environment
- Learn to make decisions that maximize a reward signal
- **Examples:**
    - Robotics (e.g., learning to walk or grasp objects)
    - Game playing (e.g., AlphaGo, poker)
    - Autonomous vehicles (e.g., learning to navigate roads)

## ML Algorithms
Do not process **Raw Data** like an image or text. Data is preprocessed through steps like ***cleaning***, ***normalization***, and ***feature extraction*** to improve the model's performance.

- **Linear Regression:**   Predicts continuous values.
- **Logistic Regression:**   Binary classification.
- **Decision Trees:**   Tree-like model for decisions.
- **k-NN (k-Nearest Neighbors):**   Classifies based on closest data points.
- **SVM (Support Vector Machines):**   Classifies by finding the best boundary.
- **Naive Bayes:**   Probabilistic classification.
- **Random Forest:**   Ensemble of decision trees.
- **K-Means:**   Unsupervised clustering algorithm.

## Limitations of ML

- Cannot directly work with ***unstructured, highly dimensional data*** (Required Feature Engineering)
- Can learn only so much from available data (i.e performance does not increase after certain threshold even if more training data is available)

# Deep Learning (DL)
- Subset of ML using Artificial Neural Networks (ANN)
- Works with large datasets and high computation
- **Example:** Image recognition, Speech recognition, self-driving cars

[Types of Neural Network](/NeuralNetwork.md)

## DL Algorithms

- **Convolutional Neural Networks (CNNs):**   Used for image processing.
- **Recurrent Neural Networks (RNNs):**   Used for sequence data (e.g., time series, language).
- **Long Short-Term Memory (LSTM):**   A type of RNN for long-term dependencies in sequences.
- **Generative Adversarial Networks (GANs):**   Used for generating data (e.g., images, videos).
- **Autoencoders:**   Used for data compression and noise reduction.
- **Transformers:**   Used for natural language processing tasks.

- Linear Models
- Neural Networks

# TensorFlow
- TensorFlow is an open-source machine learning and deep learning framework developed by Google.
- It enables building, training and deploying ML and DL models.
- Primarily used for deep learning tasks, it's versatile for various other types of machine learning.

[Explore TensorFlow Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.91235&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
