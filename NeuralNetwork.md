# Neural Network

# Types of Neural Network
### 1. Feedforward Neural Network (FNN):

- Basic neural network where data flows in one direction from input to output.
- **Applications:** Classification, regression.

### 2. Convolutional Neural Network (CNN):

- Specialized for processing grid-like data, particularly images.
- **Applications:** Image recognition, object detection, video analysis.

### 3. Recurrent Neural Network (RNN):

- Designed for sequential data, allowing information to persist over time.
- **Applications:** Time series prediction, natural language processing (NLP).

### 4. Long Short-Term Memory (LSTM):

- A type of RNN that mitigates short-term memory issues, handling long-term dependencies better.
- **Applications:** Speech recognition, text generation, sequence prediction.

### 5. Generative Adversarial Network (GAN):

- Consists of two networks (generator and discriminator) competing to produce realistic data.
- **Applications:** Image generation, deepfakes, data augmentation.

### 6. Autoencoder:

- A network trained to compress and then reconstruct data, often used for unsupervised learning.
- **Applications:** Data compression, anomaly detection.

### 7. Radial Basis Function Network (RBFN):

- Uses radial basis functions as activation functions.
- **Applications:** Function approximation, time-series prediction.

### 8. Attention Models / Transformers:

- Neural network architecture using self-attention mechanisms, widely used in NLP.
- **Applications:** Language modeling (e.g., GPT, BERT), translation, summarization.

# Activation Functions in Neural Networks

The most popular activation functions in neural networks are:

### 1. ReLU (Rectified Linear Unit)
- Output: 𝑓(𝑥)= max(0,𝑥)
- Use: Widely used in hidden layers of deep neural networks (especially CNNs).
- Reason: Simplicity, computational efficiency, and helps with faster training.

### 2. Sigmoid
- Output: 𝑓(𝑥)=1/1+𝑒−𝑥 
- Use: Common in binary classification and output layers.
- Reason: Converts inputs into probabilities between 0 and 1.

### 3. Tanh (Hyperbolic Tangent)
- Output: f(x)=tanh(x)
- Use: Often used in RNNs and LSTMs for better zero-centered outputs.
- Reason: Zero-centered, better gradient flow than sigmoid.

### 4. Leaky ReLU
- Output: f(x)=x if x>0, otherwise f(x)=0.01x
- Use: Used to overcome "dying ReLU" problem where neurons become inactive.
- Reason: Allows small gradients for negative values.

### 5. Softmax
- Output: Converts logits into probabilities across multiple classes.
- Use: Multi-class classification (output layer).
- Reason: Ensures sum of probabilities equals 1 for class predictions.

### 6. Swish
- Output: f(x)= x/1+e −x
- Use: Newer models like deep learning architectures from Google.
- Reason: Provides better performance than ReLU in some networks.
