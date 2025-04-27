# Activity: Using a Transformer (BERT) for Classification

**Goal:** See how to use a pre-trained Transformer model (like BERT) to understand text and classify its sentiment (positive/negative).

**Analogy:** We'll use a powerful text "reader" (DistilBERT) to summarize the meaning of a sentence into a list of numbers (an embedding). Then, we'll use a simple classifier (Logistic Regression) to decide if that summary indicates positive or negative sentiment.

<img src="https://jalammar.github.io/images/distilBERT/bert-distilbert-sentence-classification.png" alt="Diagram showing sentence going into BERT/DistilBERT and then into a classifier"/>

In this activity, we will use a pre-trained deep learning model to process some text. We will then use the output of that model to classify the text. The text is a list of sentences from film reviews. And we will classify each sentence as either speaking "positively" about its subject or "negatively".

## Models: Sentence Sentiment Classification

Our goal is to create a model that takes a sentence (just like the ones in our dataset) and produces either 1 (indicating the sentence carries a positive sentiment) or a 0 (indicating the sentence carries a negative sentiment). We can think of it as looking like this:

<img src="https://jalammar.github.io/images/distilBERT/sentiment-classifier-1.png" alt="Simple classifier diagram"/>

Under the hood, the model is actually made up of two models working together:

*   **Model 1: DistilBERT (The Feature Extractor):**
    This is our pre-trained Transformer (Encoder-only) model. It reads the sentence using its attention mechanism to understand the context. Its main job here is to produce a high-quality numerical representation (embedding) that captures the sentence's meaning. (DistilBERT is just a smaller, faster version of BERT).
*   **Model 2: Logistic Regression (The Classifier):**
    This is a standard, simpler machine learning model. It takes the numerical representation produced by DistilBERT and learns to map that representation to a final classification (0 or 1).

**Why two models?** This "feature extraction" approach shows how powerful embeddings from Transformers can be used as input for other models. BERT does the heavy lifting of understanding language, and the simple classifier makes the final decision based on that understanding. Another common approach is "fine-tuning", where the classification layer is integrated directly into the Transformer model itself.

The data we pass between the two models is a vector of size 768 (a list of 768 numbers). We can think of this vector as an embedding for the *entire sentence* that we can use for classification.

<img src="https://jalammar.github.io/images/distilBERT/distilbert-bert-sentiment-classifier.png" alt="Diagram showing DistilBERT outputting features to Logistic Regression"/>

## Dataset

The dataset we will use in this example is [SST2](https://nlp.stanford.edu/sentiment/index.html), which contains sentences from movie reviews, each labeled as either positive (has the value 1) or negative (has the value 0):

| sentence                                                                                                    | label |
| :---------------------------------------------------------------------------------------------------------- | :---- |
| a stirring , funny and finally transporting re imagining of beauty and the beast and 1930s horror films | 1     |
| apparently reassembled from the cutting room floor of any given daytime soap                            | 0     |
| ...                                                                                                         | ...   |

## Installing the transformers library

Let's start by installing the Hugging Face `transformers` library so we can load our deep learning NLP model (DistilBERT). You would typically run the command inside the `pip install` code block in a Python environment like Google Colab.

<details>
<summary>Click to see Installation Command</summary>

```python
# In a Colab or similar environment, run this:
# !pip install transformers torch numpy pandas scikit-learn
```

</details>

Now, let's import the necessary libraries for our script.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb # ppb is a common alias for pre-trained models from Hugging Face
import warnings
warnings.filterwarnings('ignore')
```

## Importing the dataset

We'll use the `pandas` library to read the dataset directly from a URL and load it into a dataframe structure.

<details>
<summary>Click to see Data Loading Code</summary>

```python
# Load the training data from the specified URL
# It's a TSV (Tab-Separated Values) file, so we specify the delimiter.
# It doesn't have a header row, so header=None.
df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)

print(f"Dataset loaded. Total sentences: {len(df)}")
# Display the first few rows to see the structure (Column 0 is sentence, Column 1 is label)
print("First 5 rows of the dataset:")
print(df.head()) 
```

</details>

For performance reasons and to make this activity run faster, we'll only use the first 2,000 sentences from the dataset.

```python
# Select the first 2000 rows
batch_1 = df[:2000]
print(f"\nUsing a smaller batch of {len(batch_1)} sentences.")
```

Let's check the distribution of positive (1) and negative (0) labels in our sample.

```python
# Count the occurrences of each label in the second column (index 1)
print("\nValue counts for labels in our sample (0=Negative, 1=Positive):")
print(batch_1[1].value_counts())
```

## Loading the Pre-trained BERT model (DistilBERT)

Now, we load the pre-trained Transformer model. We'll use DistilBERT, a lighter version of BERT. Remember from the summary chapter: these models have already learned a lot about language from processing huge amounts of text during their pre-training phase.

<details>
<summary>Click to see Model and Tokenizer Loading Code</summary>

```python
# Specify the model class, tokenizer class, and pre-trained weights identifier for DistilBERT
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# If you wanted to use the larger BERT model instead, you could uncomment this line:
# model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# --- Load the Tokenizer ---
# The tokenizer prepares the text in the specific way the model expects.
print(f"Loading tokenizer: {pretrained_weights}")
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

# --- Load the Model ---
# This downloads/loads the pre-trained model weights. 
# We load `DistilBertModel` which is the base model without a specific task head.
print(f"Loading model: {pretrained_weights}")
model = model_class.from_pretrained(pretrained_weights)
print("Model and tokenizer loaded.")
```

</details>

**Explanation:** We've now loaded two key components from the Hugging Face library:
*   `model`: This object holds the DistilBERT neural network architecture and its pre-trained weights, which contain its learned understanding of language.
*   `tokenizer`: This object knows how to convert text into the specific format (subword tokens, IDs, special tokens) that *this particular model* requires.

## Model #1: Preparing the Dataset for DistilBERT

Just like we discussed in the summary chapter, raw text needs processing before being fed into the model.

### Step 1: Tokenization (Matches Summary Step 1)

We convert the sentences into sequences of Token IDs using the loaded tokenizer. This includes adding special tokens like `[CLS]` at the beginning and `[SEP]` at the end.

<details>
<summary>Click to see Tokenization Code</summary>

```python
# Apply the tokenizer's encode function to each sentence in the first column (index 0)
# `add_special_tokens=True` handles adding [CLS] and [SEP]
tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

print("Tokenization complete. Example:")
# Show the first sentence and its corresponding token IDs
print(f"Original Sentence 0: {batch_1[0][0]}")
print(f"Token IDs for Sentence 0: {tokenized[0]}")
```

</details>

**Explanation:** Each sentence in our `batch_1` dataframe has been converted into a list of numbers (Token IDs). This is the numerical input the model understands.

### Step 2: Padding (Making Sequences Equal Length)

Transformer models often process data in batches for speed. To create a batch (like a table of numbers), all sequences in the batch must have the same length. We **pad** shorter sequences by adding a special padding token ID (usually 0) at the end until they match the length of the longest sequence in the batch.

<details>
<summary>Click to see Padding Code</summary>

```python
# Find the length of the longest sequence of token IDs
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)
print(f"Maximum sequence length in this batch: {max_len}")

# Create a new list where each sequence is padded with 0s to max_len
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

# Display the shape of the resulting array (number of sentences x max_len)
print(f"Shape of padded array: {padded.shape}") 
# Show the padded sequence for the first sentence
print("Example padded sequence (Sentence 0):")
print(padded[0])
```

</details>

**Explanation:** We found the longest sentence (in terms of tokens) and added 0s to the end of all other sentences so they all have that same length. `padded` is now a 2D array suitable for batch processing.

### Step 3: Masking (Ignoring Padding)

The padding tokens (0s) don't represent real words. We need to tell the model's attention mechanism to ignore them during processing. We create an **attention mask** – an array of the same shape as `padded`, containing 1s for real tokens and 0s for padding tokens.

<details>
<summary>Click to see Attention Mask Code</summary>

```python
# Create the attention mask based on the padded array
# Where `padded` is not 0, the mask value is 1; otherwise, it's 0.
attention_mask = np.where(padded != 0, 1, 0)

# Display the shape of the mask (should match `padded`)
print(f"Shape of attention mask: {attention_mask.shape}")
# Show the attention mask for the first sentence
print("Example attention mask (Sentence 0):")
print(attention_mask[0])
```

</details>

**Explanation:** The `attention_mask` clearly indicates which parts of the input the model should focus on (where the mask is 1) and which parts it should ignore (where the mask is 0).

## Model #1: Processing with DistilBERT (Deep Learning!)

With the inputs properly formatted (token IDs, attention mask), we can now feed them into the pre-trained DistilBERT model. This is where the core Transformer processing (like the Encoder Blocks and Attention discussed in the summary) happens.

<img src="https://jalammar.github.io/images/distilBERT/bert-distilbert-tutorial-sentence-embedding.png" alt="Diagram showing inputs going into BERT/DistilBERT"/>

The model will process the batch and return its outputs. We are primarily interested in the `last_hidden_state`.

<details>
<summary>Click to see Model Processing Code</summary>

```python
# Convert the numpy arrays (padded IDs and mask) into PyTorch tensors, 
# which is the format expected by the Hugging Face model.
input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

# Process the batch through the DistilBERT model.
# `torch.no_grad()` tells PyTorch not to calculate gradients, saving memory 
# and computation, as we are only doing inference (prediction), not training BERT.
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)

# The output from the base DistilBertModel is a tuple. 
# The first element contains the hidden states from the last layer.
last_hidden_states = outputs[0] 

# Display the shape of the output tensor
print(f"Shape of DistilBERT output (last_hidden_states): {last_hidden_states.shape}")
# Expected Shape: (batch_size, sequence_length, hidden_dimension)
# e.g., (2000 sentences, max_len tokens per sentence, 768 features per token)
```

</details>

**Explanation:** We passed our prepared data (token IDs and attention masks) through the DistilBERT model. The output `last_hidden_states` holds the final, context-rich embedding vector generated by the Transformer for *every* token in *every* sentence. These vectors reflect the understanding the model gained using its attention mechanism.

## Extracting the Sentence Embedding ([CLS] Token)

For sentence classification, a common technique with BERT-like models is to use the embedding of the special `[CLS]` token (which we added at the beginning of every sentence, index 0). This token's final hidden state is often treated as a summary representation of the entire sentence's meaning, suitable for classification.

<img src="https://jalammar.github.io/images/distilBERT/bert-output-tensor-selection.png" alt="Diagram highlighting the CLS token output"/>

We will extract just this first vector (`[:,0,:]`) for each sentence to use as input features for our simple Logistic Regression classifier.

<details>
<summary>Click to see Feature Extraction Code</summary>

```python
# Select the hidden state corresponding to the first token ([CLS]) for all sentences.
# `last_hidden_states[:, 0, :]` slices the tensor:
#   :   -> all sentences in the batch
#   0   -> the first token (the [CLS] token)
#   :   -> all 768 hidden features
features = last_hidden_states[:,0,:].numpy() # Convert the selected PyTorch tensor to a NumPy array

# Display the shape of our extracted features
print(f"Shape of extracted features (sentence embeddings): {features.shape}")
# Expected Shape: (Number of sentences, Hidden dimension), e.g., (2000, 768)
```

</details>

**Explanation:** We have now successfully used DistilBERT as a **feature extractor**. The variable `features` holds a single, powerful 768-dimensional vector for each of our 2000 sentences. Each vector captures the semantic essence of the sentence as understood by the Transformer.

Let's get the corresponding labels (0 or 1 for negative/positive) ready.

```python
# Extract the labels from our initial batch dataframe (column index 1)
labels = batch_1[1]
print(f"Shape of labels: {labels.shape}") # Expected: (2000,)
```

## Model #2: Logistic Regression Classifier

Now, we'll use the sentence embeddings (`features`) produced by DistilBERT to train a standard Logistic Regression classifier. This demonstrates the feature extraction approach where the complex Transformer provides input to a simpler downstream model.

### Train/Test Split

As is standard practice in machine learning, we split our extracted features and labels into a training set (to teach the model) and a testing set (to evaluate its performance on unseen data).

<details>
<summary>Click to see Train/Test Split Code</summary>

```python
# Split the features and labels into training (75%) and testing (25%) sets
# random_state ensures reproducibility if needed
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, random_state=42)

print(f"Training features shape: {train_features.shape}")
print(f"Testing features shape: {test_features.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Testing labels shape: {test_labels.shape}")
```

</details>

<img src="https://jalammar.github.io/images/distilBERT/bert-distilbert-train-test-split-sentence-embedding.png" alt="Diagram showing train/test split"/>

### [Optional Bonus] Grid Search for Parameters

*(This section searches for optimal settings for Logistic Regression. We'll skip running it to keep the activity focused, but the code shows how it could be done.)*

<details>
<summary>Click to see Optional Grid Search Code (Skipped)</summary>

```python
# This code would search for the best regularization parameter 'C'
# parameters = {'C': np.linspace(0.0001, 100, 20)}
# grid_search = GridSearchCV(LogisticRegression(), parameters)
# grid_search.fit(train_features, train_labels)

# print('best parameters: ', grid_search.best_params_)
# print('best scrores: ', grid_search.best_score_)
```

</details>

### Train the Logistic Regression Model

We train the Logistic Regression model using the sentence embeddings (`train_features`) and their known sentiment labels (`train_labels`).

<details>
<summary>Click to see Logistic Regression Training Code</summary>

```python
# Initialize the Logistic Regression model
# solver='liblinear' is often a good choice for this type of data
lr_clf = LogisticRegression(solver='liblinear')

# Train the model on the training data
lr_clf.fit(train_features, train_labels)

print("Logistic Regression model trained successfully.")
```

</details>

<img src="https://jalammar.github.io/images/distilBERT/bert-training-logistic-regression.png" alt="Diagram showing logistic regression training"/>

**Explanation:** The Logistic Regression algorithm learned to find patterns within the 768-dimensional embedding vectors that differentiate positive sentences from negative ones. It essentially learned a decision boundary in that high-dimensional space based on the training examples.

## Evaluating Model #2

Let's see how well our combined system (DistilBERT embeddings + Logistic Regression) performs on the test set (data it hasn't seen during training). Accuracy is a common metric for classification.

<details>
<summary>Click to see Evaluation Code</summary>

```python
# Evaluate the trained Logistic Regression model on the test set
accuracy = lr_clf.score(test_features, test_labels)
print(f"Accuracy on Test Set: {accuracy:.4f}")

# For comparison, let's see how a "dummy" classifier performs.
# This one simply predicts the most frequent class from the training set every time.
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy='most_frequent')
# Dummy classifier also needs to be 'fitted' to know the most frequent class
dummy_clf.fit(train_features, train_labels) 
dummy_accuracy = dummy_clf.score(test_features, test_labels)
print(f"Dummy Classifier Accuracy (predicts most frequent): {dummy_accuracy:.4f}")
```

</details>

**Explanation:** The accuracy score shows the proportion of test sentences our model classified correctly. Comparing it to the dummy classifier confirms that our approach learned meaningful patterns from the text via the DistilBERT embeddings. An accuracy significantly higher than the dummy score (which is often around 50% for balanced datasets) indicates success.

## Comparison to State-of-the-Art

How does this simple "feature extraction + logistic regression" approach compare?

*   Our accuracy (likely in the 80-85% range) is much better than random guessing.
*   However, **fine-tuning** DistilBERT directly on the SST2 dataset (adjusting the Transformer's weights for the classification task) typically achieves higher accuracy (around 90-91%).
*   Fine-tuning the larger **BERT** model achieves even better results (around 94-95%).
*   The absolute best models on this benchmark reach ~97%.

This shows that while using pre-trained embeddings as features is powerful and relatively easy, fine-tuning the Transformer model itself often yields superior performance because the entire model adapts more closely to the specific task.

## Conclusion: Your First BERT Application!

Congratulations! You've successfully walked through a common workflow using a Transformer model:

1.  **Loaded** a pre-trained Transformer (DistilBERT) and its tokenizer.
2.  **Prepared** text data using Tokenization, Padding, and Attention Masking.
3.  **Used** the Transformer as a feature extractor to generate meaningful sentence embeddings (using the `[CLS]` token's output).
4.  **Trained** a simple machine learning classifier (Logistic Regression) on these embeddings to perform sentiment analysis.
5.  **Evaluated** the performance of the combined system.

This gives you a practical feel for how Encoder models like BERT understand text and how their outputs can be leveraged for downstream tasks. You've "built" a working sentiment classifier by combining these powerful tools! The next step in learning might be to explore fine-tuning, where the Transformer model is more deeply adapted to the specific task.

---

## Ref

- [A Visual Notebook to Using BERT for the First Time](https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb)
- [BERT 101 🤗 State Of The Art NLP Model Explained](https://huggingface.co/blog/bert-101)
- [Train your tokenizer](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb)