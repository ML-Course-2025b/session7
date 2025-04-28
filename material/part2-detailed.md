## Introduction to Transformer Architectures: Core Concepts (Review Material)

### 1. The Need for Transformers: Overcoming Limitations

Large Language Models (LLMs) aim to understand and generate human language. Before the advent of Transformers, models like Recurrent Neural Networks (RNNs) and their variants (LSTMs, GRUs) were the standard for sequence processing. These models process text sequentially, token by token, maintaining an internal state or "memory" to capture context from previous tokens.

However, this sequential approach presented challenges:

*   **Limited Parallelism:** Processing tokens one after another makes it difficult to fully utilize modern parallel computing hardware (like GPUs), leading to slower training and inference times, especially on long sequences.
*   **Long-Range Dependencies:** While designed to handle sequences, RNNs can struggle to effectively connect information across very long distances in text. The influence of tokens from much earlier in the sequence can diminish over time (the "vanishing gradient" problem), making it hard to capture long-range context. Imagine reading a lengthy document word-by-word – it can be challenging to recall specific details from the very beginning by the time you reach the end.

The Transformer architecture, introduced by Vaswani et al. in the 2017 paper "Attention Is All You Need," was designed specifically to address these issues.

### 2. Transformer Innovations: Parallelism and Attention

Transformers introduced two key innovations:

*   **Parallel Processing:** Unlike RNNs, Transformers process all tokens in an input sequence simultaneously. This inherent parallelism allows them to leverage modern hardware efficiently, dramatically speeding up training.
*   **Attention Mechanism:** This is the core idea. Instead of relying solely on sequential processing and hidden states to capture context, Attention allows the model to directly calculate the relevance of every other token in the sequence when processing a specific token. It learns to selectively "pay attention" to the most important contextual information, regardless of its distance within the sequence. This mechanism proved highly effective at capturing both short-range and long-range dependencies.

Think of the Attention mechanism as being able to instantly consult an index or cross-references across an entire document (the input sequence) to understand the meaning of a specific word or phrase within its full context, all while processing the document's sections in parallel.

### 3. Preparing Text Data: Core Components

Before the main Transformer layers can process information, the raw input text must be converted into a suitable numerical format. This involves several standard pre-processing steps:

1.  **Tokenization:** Breaking the text into smaller units (tokens).
2.  **Embeddings:** Converting these tokens into numerical vectors representing their meaning.
3.  **Positional Encoding:** Injecting information about the order of tokens in the sequence.

Let's examine each step.

#### 3.1 Tokenization: From Text to Tokens

Computers require numerical input. Tokenization breaks down a string of text into a sequence of smaller units, called tokens, which are then mapped to numerical IDs.

Simple approaches like splitting by character (`c`, `a`, `t`) obscure word meaning and create very long sequences. Splitting by word results in huge vocabularies and struggles with word variations (e.g., "run", "running"), misspellings, and new words (Out-Of-Vocabulary or OOV problem).

Modern LLMs typically use **subword tokenization** algorithms like Byte Pair Encoding (BPE) or WordPiece. These algorithms are trained on large text corpora to identify frequently occurring character sequences.

*   They keep common words as single tokens (e.g., "the", "is").
*   They break down rare or complex words into meaningful sub-units (e.g., "tokenization" might become `"token", "##ization"`). The `##` prefix (used by WordPiece) often indicates that this token is part of a larger word.
*   This approach balances vocabulary size, handles OOV words by breaking them down, and preserves some relationship between related word forms.

The output of the tokenizer is a sequence of **Token IDs**, which are integers representing the tokens in its vocabulary.

**Conceptual Example (BPE):**
BPE starts with individual characters and iteratively merges the most frequent adjacent pair.

```python
# 1. Initial state: text split into characters (space often marked)
text = "hello world use newer models"
initial_chars = list(text.replace(" ", "_")) 
print("Initial Chars:", initial_chars) 
# Output: ['h', 'e', 'l', 'l', 'o', '_', 'w', 'o', 'r', 'l', 'd', '_', 'u', 's', 'e', '_', 'n', 'e', 'w', 'e', 'r', '_', 'm', 'o', 'd', 'e', 'l', 's']

# 2. Imagine BPE learns 'e' + 'r' -> 'er' is frequent. It merges them.
print("Imagine merging 'er': ['h', 'e', 'l', 'l', 'o', '_', 'w', 'o', 'r', 'l', 'd', '_', 'u', 's', 'e', '_', 'n', 'e', 'w', 'er', '_', 'm', 'o', 'd', 'e', 'l', 's']") 

# 3. It continues merging frequent pairs ('l'+'l'->'ll', 'mo'+'d'->'mod', 'mod'+'el'->'model' etc.)
print("Further merges might yield: ['h', 'e', 'll', 'o', '_', 'w', 'o', 'r', 'l', 'd', '_', 'u', 's', 'e', '_', 'n', 'e', 'w', 'er', '_', 'model', 's']")
# Each unique final token ('h', 'e', 'll', 'o', '_', ..., 'model', 's') gets assigned an ID.
```

**Practical Example (Hugging Face Tokenizer):**
Libraries provide pre-trained tokenizers specific to models.

```python
from transformers import AutoTokenizer

# Load tokenizer for bert-base-uncased model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 

text = "Transformers are powerful."

# 1. Tokenize into subword strings
tokens = tokenizer.tokenize(text)
print("Text:", text)
print("Tokens:", tokens) 
# Output: Tokens: ['transform', '##ers', 'are', 'powerful', '.'] 

# 2. Encode tokens into numerical IDs (adds special tokens like [CLS], [SEP] for BERT)
token_ids = tokenizer.encode(text) 
print("Token IDs:", token_ids) 
# Output: Token IDs: [101, 10938, 2121, 2024, 6929, 1012, 102] 

# 3. Decode IDs back to text
decoded_text = tokenizer.decode(token_ids)
print("Decoded Text:", decoded_text)
# Output: Decoded Text: [CLS] transformers are powerful. [SEP] 
```

*Note on Tokenizer Role:* The tokenizer is a crucial pre-processing step, tightly coupled with the specific LLM it was trained with. It converts text into the numerical format the model requires but is distinct from the neural network layers of the model itself.

#### 3.2 Embeddings: Representing Token Meaning

Token IDs are just numbers. **Embeddings** provide a way to represent the *meaning* of these tokens numerically. Each Token ID is mapped to a dense **vector** (a list of numbers, e.g., 768 numbers long).

Think of this as assigning coordinates to each token in a high-dimensional "meaning space". Tokens with similar meanings are expected to have vectors that are close together in this space. These vector representations are not predefined; they are **learned** during the model's training process. The model adjusts the embedding vectors to help it perform its training task (like predicting masked words or the next word).

This mapping from Token ID to vector is stored in an **Embedding Matrix** (a large lookup table). The size of these vectors (the **embedding dimension**) is a key model hyperparameter, balancing representational capacity and computational cost.

#### 3.3 Positional Encoding: Incorporating Sequence Order

Since Transformers process tokens in parallel, the core architecture doesn't inherently know the original order of the tokens. However, sequence order is often vital for meaning (e.g., "dog bites man" vs. "man bites dog").

**Positional Encoding** reintroduces this order information. A unique vector is generated for each position in the sequence (1st, 2nd, 3rd, etc.). This positional vector is then **added** element-wise to the corresponding token's embedding vector.

The resulting vector fed into the first Transformer layer thus contains information about *both* the token's meaning (from its embedding) *and* its position in the sequence. Positional encodings can be calculated using fixed mathematical functions (like sine and cosine waves) or learned as part of the model's training.

Think of the token embedding as providing the semantic coordinates (like GPS), and the positional encoding as adding a sequence number or timestamp to that location.

**Conceptual Example (Embedding Lookup & Positional Addition):**

```python
import numpy as np 

# --- Embedding Lookup ---
vocab = {"[CLS]": 0, "cat": 1, "sat": 2, "[SEP]": 3}
token_ids = np.array([0, 1, 2, 3]) # Input IDs

# Small example embedding matrix (Learned during training)
embedding_matrix = np.array([
    [0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.9, 0.1] 
])

# Look up the vector for each token ID
token_embeddings = embedding_matrix[token_ids]
print("Token IDs:", token_ids)
print("Token Embeddings (from lookup):\n", token_embeddings)

# --- Positional Encoding Addition ---
# Example positional vectors (Fixed or Learned)
position_vectors = np.array([
    [0.0, 0.1], [0.1, 0.0], [0.2, 0.1], [0.3, 0.0] 
])

# Add positional vectors to token embeddings
final_embeddings = token_embeddings + position_vectors
print("\nPositional Vectors:\n", position_vectors)
print("\nFinal Embeddings (Token Embedding + Positional Encoding):\n", final_embeddings)
# These final_embeddings are the input to the first Transformer layer.
```

#### 3.4 Scaled Dot-Product Attention: The Core Mechanism

With position-aware embeddings ready, we reach the heart of the Transformer: the **Attention mechanism**. This mechanism allows the model to weigh the importance of different tokens when processing any single token, enabling rich contextual understanding.

For each token (represented by its position-aware embedding vector), three distinct vectors are typically derived via learned linear transformations:

*   **Query (Q):** Represents the current token asking: "What context is relevant to understanding me?"
*   **Key (K):** Represents each token in the sequence advertising: "Here's the aspect of my meaning relevant for context matching."
*   **Value (V):** Represents the actual information or meaning carried by each token: "If I'm deemed relevant, here's the content I contribute."

**The Attention Calculation Steps:**

The most common form is Scaled Dot-Product Attention, which operates as follows for each Query token:

1.  **Compute Scores:** Calculate the similarity between the Query vector and all Key vectors in the sequence using the dot product: `Score = Q · Kᵀ`. A high score indicates high relevance between the Query and a Key.
2.  **Scale:** Divide the scores by the square root of the dimension of the Key vectors (`√dₖ`). This scaling helps stabilize training, preventing scores from becoming excessively large. `Scaled Score = Score / √dₖ`.
3.  **(Optional) Apply Mask:** In decoder models or specific attention patterns, a mask is applied at this stage to prevent attention to certain positions (e.g., future tokens during generation). This involves setting the corresponding scores to negative infinity.
4.  **Compute Weights (Softmax):** Apply the Softmax function to the scaled (and potentially masked) scores. This converts the scores into a probability distribution – positive values that sum to 1. These are the **attention weights**. `Weights = Softmax(Scaled Score)`. Each weight represents the proportion of attention the current token should pay to every token (including itself).
5.  **Compute Output:** Calculate the weighted sum of all Value vectors in the sequence, using the attention weights. `Output = Weights · V`. The resulting vector is the output of the attention layer for the current token – its representation is now enriched with contextual information blended from other tokens based on their calculated relevance.

**Example Walkthrough:** Consider understanding "bank" in "I sat on the river bank". The Q vector for "bank" would have a high dot product score with the K vector for "river". After scaling and softmax, "river" would receive a high attention weight. The final output vector for "bank" would be calculated as (weight_river * V_river) + (weight_I * V_I) + (weight_sat * V_sat) + ..., resulting in a representation strongly influenced by the "river" context.

**Simplified Numerical Example:**
Let's trace the calculation for Token 1, assuming 3 tokens and 2D vectors.

```python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x)) # Stable softmax
    return e_x / e_x.sum(axis=0)

# Assume Q1, K, V derived from position-aware embeddings
Q1 = np.array([1.0, 0.5]) 
K = np.array([[1.0, 0.5], [0.5, 1.0], [0.2, 0.2]]) 
V = np.array([[10, 20],   [30, 40],   [50, 60]])   

# --- Attention Calculation for Token 1 ---
# 1. Compute Scores (Q1 dot K1, Q1 dot K2, Q1 dot K3)
scores = np.dot(Q1, K.T) 
print("Step 1: Scores (Relevance of K1, K2, K3 to Q1):", scores) 
# Output: [1.25 1.   0.3 ]

# 2. Scale Scores (Assume dk=2, sqrt(dk) approx 1.41)
d_k = K.shape[1] 
scaled_scores = scores / np.sqrt(d_k)
print(f"Step 2: Scaled Scores (divide by sqrt({d_k})≈{np.sqrt(d_k):.2f}):", scaled_scores)
# Output: [0.88388348 0.70710678 0.21213203]

# (Step 3: Masking - Not applied here)

# 4. Compute Weights (Softmax of scaled scores)
weights = softmax(scaled_scores) 
print("Step 4: Attention Weights (Probabilities):", weights)
# Output: [0.41685147 0.34946358 0.23368495] 
# Interpretation: Token 1 attends ~42% to Token 1, ~35% to Token 2, ~23% to Token 3.

# 5. Compute Output (Weighted sum of Value vectors)
attn_output1 = np.sum(V * weights[:, np.newaxis], axis=0) 
print("Step 5: Attention Output for Token 1 (Contextualized Vector):", attn_output1)
# Output: [28.84836891 38.84836891] 
# This is the updated representation for Token 1, incorporating context.
```

> [!TIP]  
> More on [attention here](./Attention.md)

#### 3.5 Multi-Head Attention

Transformers enhance the attention mechanism by using **Multi-Head Attention**. Instead of performing the attention calculation just once with one set of Q, K, V matrices, they perform it multiple times in parallel, each time using *different*, independently learned linear transformations to generate the Q, K, and V vectors. Each parallel instance is called an "attention head".

This allows the model to jointly attend to information from different representation subspaces at different positions. For example, one head might learn to focus on syntactic relationships, while another focuses on semantic similarity.

The outputs from all attention heads are typically concatenated and then passed through a final linear transformation to produce the final output of the Multi-Head Attention layer, combining the diverse contextual perspectives captured by the individual heads.

---

### 4. Encoder-Only Transformer Architecture (e.g., BERT)

Having covered the core components common to most Transformers (tokenization, embeddings, positional encoding, attention), we now focus on the first major architectural pattern: the **Encoder-Only Transformer**. The most well-known example is **BERT** (Bidirectional Encoder Representations from Transformers).

#### 4.1 Objective: Deep Bidirectional Understanding

The primary goal of an encoder-only model is to build a deep, contextual understanding of an *entire* input text sequence. Unlike models that process text sequentially or generate text step-by-step, encoders aim to analyze the full context surrounding each token.

They achieve this using the **Self-Attention** mechanism in a way that allows every token to attend to every other token in the input, regardless of position (both preceding and succeeding tokens). This creates **bidirectional** context – the representation learned for any given word is informed by the words that come both before *and* after it.

Think of an encoder as an expert reader who meticulously analyzes an entire document, reading back and forth, considering all sentences together to fully grasp the meaning and role of each word within the complete text before drawing conclusions.

#### 4.2 BERT Architecture: Stacking Encoder Blocks

BERT's architecture is fundamentally a stack of identical **Encoder Blocks**. Common models like BERT-Base have 12 such blocks, while BERT-Large has 24.

**The Processing Pipeline in BERT:**

1.  **Input Text & Tokenization:** The input text is tokenized using a subword tokenizer (BERT uses WordPiece). Specific to BERT, a special `[CLS]` (classification) token is added to the beginning of the sequence, and a `[SEP]` (separator) token is added at the end of each distinct sentence (or just at the end if there's only one sentence).
2.  **Input Embeddings:** Each token ID is converted into an embedding vector. In BERT, the final input embedding for each token is the sum of three distinct embeddings:
    *   **Token Embedding:** Represents the meaning of the token itself (from the embedding matrix).
    *   **Positional Embedding:** Represents the token's position in the sequence (these are learned, not fixed functions, in BERT).
    *   **Segment Embedding:** Indicates which sentence the token belongs to (e.g., Sentence A or Sentence B), primarily used for the Next Sentence Prediction pre-training task.
3.  **Encoder Stack Processing:** The sequence of input embeddings is passed sequentially through the stack of identical Encoder Blocks. Each block refines the representations based on the global context.
4.  **Output Representations:** The final layer outputs a sequence of vectors, where each vector is the contextualized embedding for the corresponding input token (including `[CLS]` and `[SEP]`). This output captures the meaning of each token within the context of the entire input sequence.

**Inside an Encoder Block:**
Each Encoder Block performs two main operations:

1.  **Multi-Head Self-Attention:** This applies the scaled dot-product attention mechanism multiple times in parallel (using multiple "heads"). Each head learns different aspects of token relationships. Importantly, in the encoder, this attention is *bidirectional*, allowing every token to attend to every other token in the input sequence.
2.  **Position-wise Feed-Forward Network (FFN):** After the attention mechanism, the output for each token position is passed independently through a standard two-layer fully connected neural network. This FFN further processes the contextual information captured by the attention layer.

Both the Multi-Head Attention sub-layer and the FFN sub-layer within each block are followed by a **Residual Connection** (adding the sub-layer's input to its output) and **Layer Normalization**. These components are crucial for training deep networks effectively, preventing issues like vanishing gradients and stabilizing the learning process.

#### 4.3 Learning Language: BERT's Pre-training Tasks

BERT's impressive ability to understand language comes from its **pre-training** phase, where it learns general language patterns from vast amounts of unlabeled text (e.g., Wikipedia, BooksCorpus). This is achieved through two ingenious unsupervised tasks:

1.  **Masked Language Model (MLM):**
    *   **Objective:** To learn deep bidirectional context.
    *   **Process:** Before feeding a sequence into BERT, approximately 15% of its tokens are randomly selected. These tokens are then manipulated: 80% of the time they are replaced with a special `[MASK]` token, 10% of the time they are replaced with a random token from the vocabulary, and 10% of the time they are left unchanged.
    *   **Training:** The model's task is to predict the *original* identity of these masked/altered tokens, using only the surrounding (unmasked) context. Because the model can see tokens both before and after the masked position, it is forced to learn rich bidirectional representations of words and context.
2.  **Next Sentence Prediction (NSP):**
    *   **Objective:** To learn relationships between sentences.
    *   **Process:** The model receives pairs of sentences (A and B) as input, separated by the `[SEP]` token. For 50% of the pairs, sentence B is the actual sentence that followed sentence A in the original corpus. For the other 50%, sentence B is a random sentence unrelated to A.
    *   **Training:** The model uses the output representation of the `[CLS]` token to predict whether sentence B was the actual next sentence (`IsNext`) or a random one (`NotNext`).
    *   **Relevance:** While intended to help with downstream tasks requiring sentence-pair understanding (like Question Answering), the actual benefit of NSP compared to MLM alone has been debated, and some subsequent models omit it.


> [!TIP]  
> More on [BERT's Pre-training Tasks](./Pre-training-Tasks.md)


#### 4.4 BERT Model Configurations

BERT models are available in different sizes, primarily defined by:
*   **L:** Number of Encoder Blocks (layers).
*   **H:** Hidden size (dimensionality of embeddings/internal representations).
*   **A:** Number of attention heads in the Multi-Head Attention layers.

Common sizes include:
*   **BERT-Base:** L=12, H=768, A=12 (~110 million parameters)
*   **BERT-Large:** L=24, H=1024, A=16 (~340 million parameters)

Larger models generally offer better performance due to their increased capacity but require significantly more computational resources.

#### 4.5 Applying BERT: Fine-tuning and Feature Extraction

Pre-trained BERT models serve as powerful starting points for various downstream Natural Language Understanding (NLU) tasks. The two main ways to leverage BERT are:

1.  **Fine-tuning:**
    *   **Concept:** Add a small, task-specific classification layer (the "head") on top of the pre-trained BERT model. Train this combined model on a smaller dataset specifically labeled for the target task (e.g., sentiment labels, question-answer pairs). During fine-tuning, the weights of the pre-trained BERT model might be slightly adjusted ("unfrozen") or kept fixed ("frozen"), while the new task head learns from scratch.
    *   **Analogy:** Taking an individual with a strong general education (pre-trained BERT) and providing specific vocational training (fine-tuning) for a particular job (downstream task).
    *   **Typical Tasks:** Text Classification (e.g., sentiment analysis, topic classification - often uses the `[CLS]` token's output), Sequence Labeling (e.g., Named Entity Recognition - uses each token's output), Question Answering.
2.  **Feature Extraction:**
    *   **Concept:** Use the pre-trained BERT model as is, without further training, simply to generate contextual embeddings for input text. Pass the text through BERT and extract the final hidden state vectors (e.g., the `[CLS]` token's vector or a mean/max pool of all token vectors).
    *   **Usage:** These rich, context-aware vectors serve as high-quality features representing the input text's meaning. They can be fed into other machine learning models or used directly for tasks like semantic similarity search or clustering similar documents.
    *   **Analogy:** Using the generally educated individual's detailed assessment report (the embeddings) as input for a separate specialized analysis or decision-making process.

#### 4.6 BERT's Impact and Evolution

BERT's release marked a significant milestone, achieving state-of-the-art results on numerous NLU benchmarks like **GLUE** and **SQuAD**. It popularized the pre-training/fine-tuning paradigm.

Its success and open-source nature spurred considerable follow-up research, leading to variants like:
*   **RoBERTa:** Optimized pre-training strategy (more data, dynamic masking, no NSP).
*   **ALBERT:** Focused on parameter reduction for efficiency.
*   **DistilBERT:** A smaller, faster distilled version.

Furthermore, numerous **domain-specific** BERT models have been developed by further pre-training or fine-tuning on specialized corpora (e.g., `BioBERT` for biomedical text, `FinBERT` for financial text), demonstrating the adaptability of the core architecture. Task-specific fine-tuned models are also widely available (e.g., for sentiment, toxicity detection).

#### 4.7 Practical Examples with Hugging Face `transformers`

The Hugging Face `transformers` library makes using BERT models very accessible.

**Example 1: Fine-tuning for Classification (using a pre-fine-tuned model via `pipeline`)**

```python
from transformers import pipeline

# This pipeline loads a BERT-based model already fine-tuned for sentiment analysis
# It handles tokenization, inference, and output formatting.
classifier = pipeline(
    "sentiment-analysis", 
    model="nlptown/bert-base-multilingual-uncased-sentiment" # Predicts 1-5 stars
) 

result = classifier("This is a well-structured and informative course.")
print(result) 
# Output (example): [{'label': '5 stars', 'score': 0.8...}] 
```
This demonstrates using a model where the fine-tuning step has already been done for a specific task.

**Example 2: Feature Extraction (getting contextual embeddings)**

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load base pre-trained model (no task-specific head) and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

sentences = [
    "BERT provides contextual embeddings.",
    "These vectors capture meaning in context."
]

# Tokenize input
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# Get model outputs (run inference)
with torch.no_grad():
    outputs = model(**inputs)

# Extract embeddings, e.g., mean pooling of the last hidden state
# (Requires careful handling of padding tokens using the attention mask)
attention_mask = inputs['attention_mask']
mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
mean_pooled_embeddings = sum_embeddings / sum_mask

embeddings_np = mean_pooled_embeddings.numpy() 

print(f"Shape of extracted embeddings: {embeddings_np.shape}") 
# Output (example): Shape of extracted embeddings: (2, 768) 
# (2 sentences, 768 dimensions per embedding)

# These embeddings_np vectors can now be used for downstream tasks like similarity checks.
```

---

### 5. Decoder-Only Transformer Architecture (e.g., GPT)

While Encoder-Only models like BERT excel at understanding input text, another major class of Transformers focuses on **generating** text. These are the **Decoder-Only** models, with the **GPT** (Generative Pre-trained Transformer) series being the most prominent example.

#### 5.1 Objective: Autoregressive Text Generation

The primary goal of a decoder-only model is to generate coherent and contextually relevant text sequences. They operate **autoregressively**, meaning they produce the output sequence one token at a time. The prediction of each new token depends conditioned on the sequence of tokens generated previously.

Think of a decoder model as a writer composing text sequentially, where each new word written is chosen based on the specific words that have come immediately before it in the current sentence or paragraph.

#### 5.2 Key Feature: Masked Self-Attention

To operate autoregressively, a decoder must not "see into the future" when predicting the next token. It should only rely on the tokens generated *up to the current position*. Standard bidirectional self-attention (used in encoders) would violate this causality principle.

Decoder-only models solve this using **Masked Self-Attention** (also called Causal Self-Attention):

*   **Mechanism:** During the self-attention calculation (specifically, before the Softmax step), a mask is applied to the attention scores. This mask prevents any token from attending to subsequent tokens in the sequence. Scores corresponding to attention between token `i` and any token `j` where `j > i` are effectively set to negative infinity.
*   **Effect:** After the Softmax function, these masked positions receive an attention weight of zero. This ensures that the representation calculated for a token at position `i` is solely influenced by tokens at positions 1 to `i`, maintaining the left-to-right, step-by-step generative process.

This masking is the fundamental difference enabling decoders to generate sequences one token after another, based only on past context.

#### 5.3 GPT Architecture and Generation Process

A typical GPT-style model consists of a stack of identical **Decoder Blocks**.

**Inside a Decoder Block:**
Each block usually contains:

1.  **Masked Multi-Head Self-Attention:** Applies the causal attention mechanism described above, allowing each token to attend only to itself and preceding tokens in the sequence, using multiple heads in parallel.
2.  **Add & Norm:** Residual connection followed by Layer Normalization.
3.  **Position-wise Feed-Forward Network (FFN):** A standard two-layer fully connected network applied independently to each token's representation.
4.  **Add & Norm:** Another residual connection and Layer Normalization.

*(Note: In contrast to decoder blocks within an Encoder-Decoder model, these Decoder-Only blocks lack the Cross-Attention layer, as there is no separate encoder output to attend to).*

**The Autoregressive Generation Workflow:**
Generating text with a decoder model like GPT involves a repetitive loop:

1.  **Start with a Prompt:** Provide an initial sequence of text (the prompt) to guide the generation, e.g., "Artificial intelligence is".
2.  **Tokenize and Embed:** Convert the prompt into Token IDs and then into position-aware embedding vectors (using token embeddings and positional encodings).
3.  **Process through Decoder Stack:** Pass these embedding vectors through the stack of Decoder Blocks. The output corresponding to the *last* token of the current sequence is of particular interest.
4.  **Calculate Next Token Logits:** Transform the final hidden state vector of the last token into a vector of scores (logits) over the entire vocabulary. Each score represents the model's predicted likelihood for that vocabulary token being the *next* token in the sequence.
5.  **Convert Logits to Probabilities:** Apply the Softmax function to the logits to obtain a probability distribution across all possible next tokens.
6.  **Sample the Next Token:** Select the next token based on this probability distribution. Several strategies exist:
    *   *Greedy Sampling:* Always choose the token with the highest probability. (Deterministic but can be repetitive).
    *   *Top-K Sampling:* Randomly sample from the `K` tokens with the highest probabilities. (Introduces randomness).
    *   *Top-P (Nucleus) Sampling:* Randomly sample from the smallest set of top tokens whose cumulative probability exceeds a threshold `P`. (Balances quality and diversity).
7.  **Append and Repeat:** Append the ID of the sampled token to the current sequence. This extended sequence becomes the input for the next iteration. Repeat steps 3-7 to generate the subsequent token, continuing until a desired sequence length is reached or a special end-of-sequence (EOS) token is generated.

#### 5.4 Common Use Cases for Decoder-Only Models

Their generative nature makes decoder-only models suitable for tasks such as:

*   **Open-Ended Text Generation:** Writing stories, articles, poetry, dialogue.
*   **Chatbots and Conversational AI:** Generating human-like responses.
*   **Code Generation:** Assisting programmers by generating code snippets or functions.
*   **Summarization, Translation, Question Answering (via Prompting):** While specialized Encoder-Decoder models exist, decoders can perform these tasks effectively when given appropriate instructions or examples directly within the input prompt (known as in-context learning, zero-shot, or few-shot prompting). For instance, prefixing text with "Summarize this:" can prompt the model to generate a summary.

#### 5.5 Practical Example: Text Generation with Hugging Face

The `transformers` library provides easy access to pre-trained decoder models for text generation.

**Example: Generating text using GPT-2**

```python
from transformers import pipeline

# Load the text-generation pipeline, often defaults to GPT-2 or allows model specification.
generator = pipeline("text-generation", model="gpt2") 

prompt = "Machine learning is a field of study focused on"
print(f"Input Prompt: {prompt}")
print("-" * 20)

# Generate text based on the prompt
# max_length defines the total length of the output (prompt + generated tokens)
generated_sequences = generator(
    prompt, 
    max_length=70,  
    num_return_sequences=1 # Number of different sequences to generate
    # Sampling parameters like do_sample=True, top_k=50 can be added for more diverse output
)

# Display the result
for i, seq in enumerate(generated_sequences):
    print(f"Generated Sequence {i+1}:\n{seq['generated_text']}")

# Example Output (will vary due to model/sampling):
# Generated Sequence 1:
# Machine learning is a field of study focused on the development of computer systems 
# that can learn from and make decisions on data. Machine learning algorithms build a 
# mathematical model based on sample data, known as "training data", in order to make 
# predictions or decisions without being explicitly programmed to do so.

# Explanation: The pipeline tokenized the prompt, fed it through the GPT-2 model, 
# and then iteratively predicted and appended the next token using the autoregressive 
# process until the max_length limit was approached.
```

---

### 6. Encoder-Decoder Transformer Architecture (e.g., T5, BART)

Beyond models focused solely on understanding (Encoders) or generation (Decoders), the third primary architecture is the **Encoder-Decoder Transformer**, also known as a **Sequence-to-Sequence (Seq2Seq)** model. This was the architecture presented in the original "Attention Is All You Need" paper and is employed by models like T5, BART, and many systems designed for machine translation (e.g., MarianMT).

#### 6.1 Objective: Mapping Input Sequences to Output Sequences

The fundamental goal of an Encoder-Decoder model is to transform or map an entire input sequence into a potentially different output sequence. This makes it naturally suited for tasks where the input and output formats differ significantly, such as translating between languages or condensing a long document into a short summary.

#### 6.2 Architecture: Combining Encoder and Decoder Stacks

As the name implies, this architecture explicitly includes two main components:

1.  **Encoder Stack:** This part processes the *entire* input sequence using bidirectional self-attention layers, similar to BERT. Its role is to create a comprehensive contextual representation (a set of hidden state vectors) of the source sequence.
2.  **Decoder Stack:** This part generates the *output* sequence autoregressively (token by token), similar to GPT. It uses masked self-attention to ensure it only considers the output tokens generated so far.

#### 6.3 The Bridge: Cross-Attention

The critical element connecting the Encoder and Decoder is the **Cross-Attention mechanism**, located within each block of the Decoder stack (typically following the masked self-attention layer).

*   **Mechanism:** In Cross-Attention:
    *   The **Query (Q)** vectors are derived from the Decoder's *own* hidden states (representing the output sequence generated up to the current step).
    *   The **Key (K)** and **Value (V)** vectors are derived from the *final hidden state outputs of the Encoder stack*.
*   **Function:** This allows the Decoder, at each step of generating an output token, to look back and focus its attention on the most relevant parts of the *entire encoded input sequence*. It effectively asks: "Based on the output I've generated so far (Decoder Query), which parts of the original source information (Encoder Keys/Values) are most helpful for predicting the very next output token?"

Think of this like a translator (the Decoder) who continually glances back at the original source text (the Encoder output) while composing the translation sentence by sentence.

#### 6.4 Common Use Cases

The Encoder-Decoder architecture is highly effective for tasks inherently involving sequence transformation:

*   **Machine Translation:** Translating text from a source language to a target language.
*   **Summarization:** Generating a concise summary from a longer text document.
*   **Generative Question Answering:** Providing context and a question as input and generating a free-form textual answer as output.
*   **Text Style Transfer or Normalization:** Converting text from one style to another (e.g., formal to informal) or correcting grammatical errors.

#### 6.5 Practical Example: Summarization with T5

The Hugging Face `pipeline` can be used with Encoder-Decoder models like T5 for tasks like summarization.

```python
from transformers import pipeline

# Load the summarization pipeline with a T5 model (e.g., t5-small)
summarizer = pipeline("summarization", model="t5-small") 

long_text = (
    "The Transformer architecture relies on self-attention for parallel processing and capturing "
    "long-range dependencies, outperforming older recurrent models. Encoder-only models like BERT "
    "excel at understanding input via bidirectional context. Decoder-only models like GPT focus on "
    "autoregressive text generation using masked self-attention. Encoder-decoder models, such as "
    "T5, leverage both components along with cross-attention for sequence-to-sequence tasks."
)
print("Original Text (Input to Encoder):\n", long_text)
print("-" * 20)

# Perform summarization
summary = summarizer(
    long_text, 
    max_length=40, # Maximum length of the generated summary
    min_length=10, # Minimum length of the generated summary
    do_sample=False # Use deterministic output
)

print("Generated Summary (Output from Decoder):\n", summary[0]['summary_text'])
# Example Output: encoder-only models like BERT excel at understanding input via bidirectional context . 
# decoder-only models like GPT focus on autoregressive text generation . encoder-decoder models leverage both .

# Explanation: The T5 model's encoder processed the long text. Its decoder then generated the 
# summary token-by-token, using masked self-attention on the summary-so-far and cross-attention 
# to consult the encoded representation of the original long text.
```

For a deeper dive into the original Encoder-Decoder implementation, studying the "Attention Is All You Need" paper and the code walkthrough provided by "The Annotated Transformer" is highly recommended.

### 7. Conclusion and Further Learning

#### 7.1 Summary of Transformer Architectures

We have explored the three primary architectures based on the Transformer mechanism:

*   **Encoder-Only (e.g., BERT):** Designed for deep understanding of input text using bidirectional self-attention. Excels at tasks like classification, feature extraction, and sequence labeling.
*   **Decoder-Only (e.g., GPT):** Designed for generating text autoregressively using masked (causal) self-attention. Ideal for text generation, chatbots, and prompt-based tasks.
*   **Encoder-Decoder (e.g., T5, Original Transformer):** Designed for mapping input sequences to output sequences. Uses bidirectional self-attention in the encoder, and both masked self-attention and cross-attention (attending to encoder output) in the decoder. Suited for translation, summarization, and other sequence-to-sequence tasks.

<img src="./img/transformer.svg" width="50%">

*(See: [ General Transformer architecture](https://huggingface.co/learn/llm-course/chapter1/4#general-transformer-architecture))*

#### 7.2 Key Takeaways

*   The **Attention mechanism** is the cornerstone of Transformers, enabling parallel processing and effective capture of contextual dependencies by allowing tokens to directly weigh the importance of other tokens.
*   Fundamental pre-processing steps – **Tokenization**, **Embeddings**, and **Positional Encoding** – are essential for preparing text data for these powerful models.

#### 7.3 Beyond Transformers

While Transformer-based models currently represent the state-of-the-art and are the primary focus of most current Large Language Model (LLM) development, active research continues into alternative and complementary architectures. These efforts often aim to address challenges like the computational cost associated with Transformers' attention mechanism, especially for very long sequences, or to explore different ways of capturing dependencies.

Some notable directions include:

*   **Structured State Space Models (SSMs):** Architectures like Mamba use state-space representations, drawing inspiration from classical control theory, as an alternative to attention for modeling long-range dependencies, potentially offering computational advantages.
*   **Fourier-Based Methods:** Approaches such as the Fourier Neural Operator (FNO) operate in the frequency domain using Fourier transforms. While often applied to modeling physical systems, related ideas are explored for sequence modeling, aiming for efficiency gains.
*   **Improved Recurrent Architectures:** Research also revisits and attempts to enhance purely recurrent models (like LSTMs/RNNs) to better handle long sequences and improve parallelism.
*   **Hybrid Approaches:** Methods combining elements from different architectures (e.g., blending Fourier techniques with deep learning components) are also being investigated.

Although these alternative architectures show promise and may become more prominent, **understanding the Transformer remains crucial** for navigating the current LLM landscape, as it forms the foundation for the vast majority of today's most capable and widely used models.


#### 7.4 Resources for Further Study

*   **Foundational Paper:** Vaswani, A., et al. (2017). "Attention Is All You Need". (Available on arXiv and NeurIPS proceedings).
*   **Implementation Guide:** "The Annotated Transformer" - Harvard NLP group's detailed PyTorch code walkthrough of the original paper: [http://nlp.seas.harvard.edu/annotated-transformer/](http://nlp.seas.harvard.edu/annotated-transformer/)
*   **Practical Video Example:** Andrej Karpathy's "Let's build GPT": [Let's reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU) (Provides excellent insight into building a decoder-only model from scratch).
*   **Model & Library Hub:** Hugging Face (https://huggingface.co/): A central resource for accessing pre-trained models, datasets, tutorials, and the widely used `transformers` library.
*   **HuggingFace LLM Course:** https://huggingface.co/learn/llm-course
*   **HuggingFace AI Agents Course:** https://huggingface.co/learn/agents-course/
