# Mechanistic Interpretability: TinyStories Model/Dataset

## About

Mechanistic Interpretability: Analysing Tiny Stories models attention, MLP and embeddings.

## Findings (TinyStories-1L-21M)

Prompt:
```One day, Lucy asks Tom: "I am looking for a banana but I can't find it". Tom says: "Don't```

[Results](https://docs.google.com/spreadsheets/d/1WVAhf0EX5YMSBgO_oIZO3crwaR0GnPcI6ERkNzAJfwE/edit?usp=sharing)

### Preliminary Breakdown of Attention Heads in Transformer Block

| Head | Description |
| ---- | ----------- |
| 0 | Attends to dialogue structure (e.g. speaker changes), contextual relationships (e.g. "I" to "but", action to result), and speech delimiters. |
| 1 | Attends to start of preceding structural pivot (e.g., start of text token, comma, and full stop). For the word "but," only attends to itself, not preceding structural pivot like others. |
| 2 | Strongly attends to start of preceding structural pivot (e.g., start of text token, comma, and full stop). For the word "but," only attends to itself, not preceding structural pivot like others. |
| 3 | ~~Subject of sentence? |
| 4 | Attends to the current token / also sometimes the previous token. |
| 5 | Attends to the start of a clause and the start of a co-ordinating junction? |
| 6 | Attends to the previous token (aside from the closing speech mark, attends to the starting speech mark instead). |
| 7 | Attending to backward-related parts of skip n-grams? |
| 8 | Attends to the current token. |
| 9 | ~Attends to current and previous proper nouns? |
| 10 | ~Attends to all of the other tokens within same clause/sentence by varying amounts. |
| 11 | Attends to the start of a clause? Attends to the starting speech mark within speech. |
| 12 | ~Attending to the previous token unigram / bigram. |
| 13 | ~~Subject of the sentence? |
| 14 | Attends to narrative structures and relationships. Start of text token, commas, attends to prior proper nouns on speech marks, the word "asks" i.e., verb before proper noun "Tom," attending to "banana" on token "find." Attention head is trying to keep a high-level summary of ongoing narrative, acting as a guide when generating or predicting subsequent tokens? |
| 15 | Attending to the start of a clause? |

### Preliminary Analysis of MLP Layer in Transformer Block

Prompt: "Jack and Lily liked to watch the moon at night. They noticed that the moon changed its shape every night. Sometimes the moon was big and round, and sometimes it was"

#### 4096 MLP: blocks.0.mlp.hook_post (PreLN, GeLU)
Mean: -0.0483 \
Standard Deviation: 0.2245 \
Min Value: -0.1700 \
1st Quartile (25th Percentile): -0.1525 \
Median (50th Percentile): -0.1033 \
3rd Quartile (75th Percentile): -0.0296 \
Max Value: 11.3458 

#### 1024 MLP: blocks.0.hook_mlp_out (PreLN, GeLU)
Mean: 0.0035 \
Standard Deviation: 0.6573 \
Min Value: -2.2938 \
1st Quartile (25th Percentile): -0.2691 \
Median (50th Percentile): -0.0122 \
3rd Quartile (75th Percentile): 0.2459 \
Max Value: 55.2631 

#### Final Embed: ln_final.hook_normalized (PostLN)
Mean: -0.0357 \
Standard Deviation: 1.4694 \
Min Value: -7.1327 \
1st Quartile (25th Percentile): -0.9617 \
Median (50th Percentile): -0.0358 \
3rd Quartile (75th Percentile): 0.8951 \
Max Value: 14.2448

It seems like the 4096-dim MLP creates a foundational, more concentrated representation of the
information which the 1024-dim MLP then diversifies and possibly disentangles further. This
process might be essential to allow subsequent layers or blocks in the model to effectively
work with a broader range of information.

<!--
## Analysis Plan

**To analyze the MLP layer after attention in the transformer block of GPT-2, you'd want to examine how the model transforms the activations output by the attention mechanism, and how these are used in the subsequent feed-forward layer.**

**Given the hooks you've set, and your intention to investigate the MLP layer following the attention mechanism, you'll want to look into these specific hooks:**

- `'blocks.0.hook_attn_out'`: This is the output of the attention mechanism. This tensor would be the starting point for our analysis as it represents the aggregated context from attention heads.

- `'blocks.0.mlp.hook_pre'`: Represents the activations after the input is passed through the first layer of the MLP but before the activation function (like GELU in GPT-2).

- `'blocks.0.mlp.hook_post'`: Represents the activations after the activation function is applied.

- `'blocks.0.hook_mlp_out'`: This is the output of the MLP layer.

**Steps to Analyze the MLP Activations:**

**Basic Statistics:**
- Compute mean, variance, and other basic statistics for each of the above hooks. Are there any noticeable differences before and after the MLP layer?

**Visualization:**
- Plot heatmaps of the activations (you can do this for a batch of tokens or individual tokens).
- Compare the difference between `'hook_attn_out'` and `'hook_mlp_out'`. Look for significant changes or patterns.

**Distribution Analysis:**
- Plot histograms to see the distribution of activations. Is there a shift in the distribution after passing through the MLP? Are activations more spread out or concentrated?

**Dimensionality Reduction:**
- Use PCA or t-SNE to visualize the activations in a reduced space. This can help you identify any clusters or patterns in the activations.

**Relationship with Attention Heads:**
- As you have analyzed attention patterns, try to correlate specific attention behaviors with patterns in MLP activations. For instance, if a particular token had strong attention from a certain head, how does its activation change in the MLP?

**Activation Dynamics:**
- Analyze how different the activations are for different input samples. For instance, given two different sentences, how does the distribution of activations in `'hook_mlp_out'` differ?

**High Activation Tokens:**
- Identify tokens that have particularly high (or low) activations in `'hook_mlp_out'`. Are there specific patterns or token types that consistently result in high activations?

**Ablation Study:**
- A more advanced analysis could involve nullifying certain activations (setting them to zero) and observing the impact on the model's output. This could give insights into the importance of specific activations.

**Neuron-Level Analysis:**
- Dive deeper into individual neurons or positions in the MLP. Are there neurons that activate particularly strongly for certain input patterns?

**Lastly, remember that analyzing neural network activations is more of an art than an exact science. The above steps provide heuristics and might not lead to definite conclusions. Still, they can offer valuable insights into the model's behavior.**
-->