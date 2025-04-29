# Part 2: How Transformers Understand Text (The BERT Example - Simplified)

### Introduction: Reading Like a Transformer

Imagine understanding language like quickly scanning a whole page, instantly grasping connections between words, rather than reading word-by-word. That's the core idea behind **Transformers**, a powerful AI architecture. Older models struggled with speed and remembering context in long texts. Transformers overcome this using:

1.  **Parallel Processing:** Looking at many words at once.
2.  **Attention Mechanism:** Smartly focusing on relevant words to understand context, no matter how far apart they are.

We'll explore this using **BERT** as our main example. BERT is designed specifically for *understanding* text (it's an **Encoder-only** model). Let's see how it processes a sentence.

### The Journey of Text Through BERT

**Input:** "Cats sleep soundly."

**Step 1: Tokenization - Chopping Up Words**

Computers need numbers. **Tokenization** breaks text into smaller pieces (tokens), often "subwords," and gives each piece a number (ID).

*   **Why?** Handles new words ("soundly" might become `"sound"`, `"##ly"`) and keeps the vocabulary size reasonable. Special tokens like `[CLS]` (start) and `[SEP]` (end) are often added.

<details>
<summary>Click to see Demo: Tokenization</summary>

```python
# Make sure transformers is installed: !pip install transformers
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
text = "Cats sleep soundly."

# See the subword pieces
tokens = tokenizer.tokenize(text) 
print(f"Text: '{text}'")
print(f"Tokens: {tokens}") 

# Get the numerical IDs (adds [CLS] and [SEP])
token_ids = tokenizer.encode(text) 
print(f"Token IDs: {token_ids}") 

# Example Output:
# Text: 'Cats sleep soundly.'
# Tokens: ['cats', 'sleep', 'sound', '##ly', '.']
# Token IDs: [101, 7851, 4431, 3910, 1026, 1012, 102] 
```

*   **Explanation:** The code uses a standard BERT tokenizer to split the text into tokens (like `'cats'`, `'sleep'`, `'sound'`, `'##ly'`, `'.'`) and then converts them into unique ID numbers. This is the first step.
</details>

> [!TIP]  
> You can use [this tool](https://platform.openai.com/tokenizer) to understand how a piece of text might be tokenized by a language model


**Step 2: Embeddings - Giving Tokens Initial Meaning Vectors**

The Token IDs need meaning. An **Embedding** layer acts like a dictionary, mapping each ID to a list of numbers called a **vector**.

*   **Analogy:** Like giving each token starting coordinates on a "meaning map".
*   **Learned:** These vectors are learned during training.

<details>
<summary>Click to see Demo: Embedding Lookup (Conceptual)</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define 3D semantic embeddings (manually set for illustration)
# Format: [petness, animalness, cityness]
embedding_dict = {
    "cat":  [2, 3, 0],
    "dog":  [2.5, 2.8, 0.1],   # Close to cat
    "lion": [0, 5, 0],
    "NYC":  [0, 0, 5],         # High on cityness, unrelated to animals
}

# Simulate an embedding table and token IDs
# Create dummy table for 100 tokens with 3D vectors
vocab_size = 100
embedding_dim = 3
embedding_table = np.random.rand(vocab_size, embedding_dim)

# Add custom tokens into this table
custom_tokens = ["cat", "dog", "lion", "NYC"]
custom_ids = [10, 25, 3, 99]  # Simulated token IDs
for token, idx in zip(custom_tokens, custom_ids):
    embedding_table[idx] = embedding_dict[token]

# Retrieve embeddings using token IDs
sample_ids = custom_ids
initial_embeddings = [embedding_table[id_] for id_ in sample_ids]

# Print information similar to original code
print(f"Sample Token IDs: {sample_ids}")
initial_embeddings_np = np.array(initial_embeddings)
print(f"Shape of Initial Embeddings: {initial_embeddings_np.shape}")
print("First embedding vector:\n", initial_embeddings[0])

# Plot the 3D embeddings
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs = initial_embeddings_np[:, 0]  # petness
ys = initial_embeddings_np[:, 1]  # animalness
zs = initial_embeddings_np[:, 2]  # cityness

ax.scatter(xs, ys, zs, color='green', s=60)

# Annotate each point
for i, label in enumerate(custom_tokens):
    ax.text(xs[i], ys[i], zs[i], label, fontsize=10)

# Set axis labels
ax.set_xlabel("Petness")
ax.set_ylabel("Animalness")
ax.set_zlabel("Cityness")
ax.set_title("3D Semantic Embedding Visualization")

plt.tight_layout()
plt.show()
```

*   **Explanation:** We created a conceptual lookup table (`embedding_table`). We took our list of Token IDs and looked up the corresponding vector (list of numbers) for each ID. These vectors hold the initial "meaning".
</details>

<img src="./img/embedding.png" width="50%">

**Step 3: Positional Encoding - Adding Order Information**

Transformers look at words simultaneously, losing the original order. **Positional Encoding** adds information about *where* each token is in the sequence.

*   Another vector, representing the *position* (1st, 2nd, 3rd...) of each token, is created.
*   This position vector is **added** element-wise to the token's meaning vector (embedding).
*   **Result:** The final vector for each token now contains information about both *what* it means and *where* it is in the sequence.

<details>
<summary>Click to see Demo: Adding Positional Encoding (Conceptual)</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define semantic embeddings (manually)
embedding_dict = {
    "cat":  [2, 3, 0],
    "dog":  [2.5, 2.8, 0.1],
    "lion": [0, 5, 0],
    "NYC":  [0, 0, 5],
}
custom_tokens = ["cat", "dog", "lion", "NYC"]
custom_ids = [10, 25, 3, 99]

# Create a random embedding table and override the target embeddings
vocab_size = 100
embedding_dim = 3
embedding_table = np.random.rand(vocab_size, embedding_dim)
for token, idx in zip(custom_tokens, custom_ids):
    embedding_table[idx] = embedding_dict[token]

# Look up embeddings
sample_ids = custom_ids
initial_embeddings = np.array([embedding_table[id_] for id_ in sample_ids])
print(f"Shape of Initial Embeddings: {initial_embeddings.shape}")
print("First embedding vector:\n", initial_embeddings[0])

# Positional encoding (simplified sinusoidal method for 3D)
num_tokens = initial_embeddings.shape[0]
position_vectors = np.zeros_like(initial_embeddings)
for i in range(num_tokens):
    for j in range(embedding_dim):
        if j % 2 == 0:
            position_vectors[i, j] = np.sin(i / (10000 ** (j / embedding_dim)))
        else:
            position_vectors[i, j] = np.cos(i / (10000 ** (j / embedding_dim)))

# Combine original embeddings with position info
position_aware_embeddings = initial_embeddings + position_vectors

print(f"Shape after adding position: {position_aware_embeddings.shape}")
print("First Position-Aware Embedding Vector:\n", position_aware_embeddings[0])

# 3D Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Original positions (for reference)
ax.scatter(initial_embeddings[:, 0], initial_embeddings[:, 1], initial_embeddings[:, 2],
           color='blue', label='Original', s=50)

# Position-aware versions
ax.scatter(position_aware_embeddings[:, 0], position_aware_embeddings[:, 1],
           position_aware_embeddings[:, 2], color='red', label='With Position', s=50)

# Draw arrows to show shift due to position
for i in range(num_tokens):
    ax.plot([initial_embeddings[i, 0], position_aware_embeddings[i, 0]],
            [initial_embeddings[i, 1], position_aware_embeddings[i, 1]],
            [initial_embeddings[i, 2], position_aware_embeddings[i, 2]],
            color='gray', linestyle='--')

# Annotate
for i, token in enumerate(custom_tokens):
    ax.text(position_aware_embeddings[i, 0], position_aware_embeddings[i, 1],
            position_aware_embeddings[i, 2], f"{token}", fontsize=9)

ax.set_xlabel("Petness")
ax.set_ylabel("Animalness")
ax.set_zlabel("Cityness")
ax.set_title("3D Semantic Embeddings with Positional Encoding")
ax.legend()
plt.tight_layout()
plt.show()
```

*   **Explanation:** We created unique vectors based on position and simply added them to the meaning vectors. Now, each vector entering the main Transformer layers knows both the token's meaning *and* its position.

*(See: [Positional Encoding](./position.md))*

</details>



**Step 4: Encoder Blocks - Processing with Attention**

These position-aware vectors now go through a stack of **Encoder Blocks** (e.g., 12 in BERT-base). The key process inside is **Self-Attention**.

*   **What is Self-Attention?** It lets each word look at all other words *in the same sentence* to figure out which ones are most relevant for understanding its own meaning *in this specific context*.
*   **How (The Q/K/V Intuition):**
    *   Each word creates a **Query** (Q - "Who is relevant to me?").
    *   Each word creates a **Key** (K - "Here's what context I offer.").
    *   Each word creates a **Value** (V - "Here's my actual meaning/content.").
    *   Queries are compared to Keys to find relevance (attention scores/weights).
    *   The final output for each word is a blend (weighted average) of all words' Values, based on those attention weights.
*   **Bidirectional:** Because words look both forwards and backwards, BERT gets a deep, bidirectional understanding.

<img src="./img/qvn.png" width="50%">

<details>
<summary>Click to see Demo: The *Effect* of Attention (Simplified)</summary>

```python

```

*   **Explanation:** This demo skips the complex Q/K matching. It shows the *result* of attention: how pre-calculated relevance weights (here, `[0.3, 0.6, 0.1]` for Word 2) are used to create a new, context-blended vector (`[0.35, 0.65]`) for Word 2 by taking a weighted average of all the word's Value vectors. The model does this for *every* word. After attention, a standard **Feed-Forward Network** processes each resulting vector individually.
</details>

<img src="./img/qvn2.png" width="50%">

*(See: [Attention Mechanism](./Attention.md))*

**Step 5: Stacking Blocks**

The output from one Encoder Block becomes the input for the next. Repeating this (e.g., 12 times) allows BERT to build very deep, complex understanding.

### How BERT Learns: Pre-training (Fill-in-the-Blanks)

Where do the "right" attention weights or vector meanings come from? BERT learns them during **pre-training** on billions of sentences, mostly via **Masked Language Modeling (MLM)**.

*   **Concept:** It's like a massive "fill-in-the-blanks" game.
*   **Process:** The model sees sentences with random words hidden (`[MASK]`). It must predict the original word using the surrounding context (words before *and* after).

<details>
<summary>Click to see Demo: Creating Masked Input for MLM</summary>

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
text = "The cat sat on the [MASK]." # We manually masked 'mat'

# Get the token IDs for this masked input
masked_token_ids = tokenizer.encode(text)

print(f"Original Text (with mask): {text}")
print(f"Token IDs for Masked Input: {masked_token_ids}")

# Find the ID for the word 'mat'
mat_token_id = tokenizer.convert_tokens_to_ids('mat')
print(f"(BERT's goal in training would be to predict ID {mat_token_id} for the mask)") 
```

*   **Explanation:** This shows what the input for the MLM task looks like. By trying to predict the masked word millions of times, the model learns grammar, context, and how words relate, tuning its attention mechanism and embedding meanings. (BERT also used Next Sentence Prediction, but MLM is the core idea).
</details>



### BERT's Output and Use

After passing through all Encoder blocks, BERT outputs final **contextualized embeddings** – one vector (list of numbers) for each input token. These vectors are rich with meaning derived from the full sentence context.

<details>
<summary>Click to see Demo: Getting Final Output Embeddings</summary>

```python
from transformers import AutoTokenizer, AutoModel # Use AutoModel for base embeddings
import torch # Still need torch slightly for model loading/inference with transformers

# Use a smaller BERT-like model for faster loading/running if possible
# Or use 'bert-base-uncased'
model_name = 'bert-base-uncased' 
# model_name = 'prajjwal1/bert-mini' # Uncomment to try smaller model if installed

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print(f"Loaded model: {model_name}")
except Exception as e:
    print(f"Error loading {model_name}, try installing or check name: {e}")
    model = None

if model:
    text = "Cats sleep soundly."
    # Prepare input for the model using the tokenizer
    inputs = tokenizer(text, return_tensors='pt') # 'pt' returns PyTorch tensors

    # Get the final outputs from the model
    # torch.no_grad() avoids tracking computations, making it faster
    with torch.no_grad(): 
        outputs = model(**inputs)

    # Extract the final hidden states (the contextual embeddings)
    last_hidden_states = outputs.last_hidden_state 

    # Convert the output tensor to a numpy array for easier viewing without PyTorch details
    final_embeddings_np = last_hidden_states.cpu().numpy()

    print(f"\nInput text: '{text}'")
    print(f"Shape of Final Output Embeddings: {final_embeddings_np.shape}") 
    # Shape: (batch_size=1, num_tokens_in_sequence, embedding_dimension)
    # e.g., (1, 6, 768) for bert-base
    # print("Final embedding vector for the first token ([CLS]):\n", final_embeddings_np[0, 0, :]) # Uncomment to view a vector
```

*   **Explanation:** We passed our tokenized text through the BERT model. The output contains the final, context-rich vector for each input token. These are much more informative than the initial embeddings.
</details>



These final embeddings are highly useful for tasks requiring text understanding:
*   **Classification:** Add a simple classifier on top (often using the `[CLS]` token's output vector).
*   **Feature Extraction:** Use these vectors as input for other ML models or for clustering.

### Summary: The Big Picture

BERT uses stacked **Encoder Blocks**, featuring **Self-Attention (Q/K/V)**, to read text bidirectionally and create deep contextual understanding. It learns this ability through **Pre-training** (like MLM). Its final output is rich **contextual embeddings**, great for analysis tasks.

This Encoder model is one key type of Transformer. Others, like Decoders (GPT) or Encoder-Decoders (T5), build on these core ideas for different tasks like text generation or translation.

<img src="./img/transformer.svg" width="50%">

*(See: [ General Transformer architecture](https://huggingface.co/learn/llm-course/chapter1/4#general-transformer-architecture))*

---

> [!NOTE]  
> For more details on specific components or other architectures,[ please refer to this reading](./part2-detailed.md).