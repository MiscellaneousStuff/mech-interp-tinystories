# Mechanistic Interpretability: TinyStories Model/Dataset

## About

Mechanistic Interpretability: Analysing Tiny Stories models attention, MLP and embeddings

## Findings

Prompt:
```One day, Lucy asks Tom: "I am looking for a banana but I can't find it". Tom says: "Don't```

[Results](https://docs.google.com/spreadsheets/d/1WVAhf0EX5YMSBgO_oIZO3crwaR0GnPcI6ERkNzAJfwE/edit?usp=sharing)

### Preliminary Breakdown of Attention Heads

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
| 10 | ? |
| 11 | Attends to the start of a clause? Attends to the starting speech mark within speech. |
| 12 | ~Attending to the previous token unigram / bigram. |
| 13 | ~~Subject of the sentence? |
| 14 | Attends to narrative structures and relationships. Start of text token, commas, attends to prior proper nouns on speech marks, the word "asks" i.e., verb before proper noun "Tom," attending to "banana" on token "find." Attention head is trying to keep a high-level summary of ongoing narrative, acting as a guide when generating or predicting subsequent tokens? |
| 15 | Attending to the start of a clause? |