# Amazon Reviews Analytics and Recommendation Systems

This project analyzes Amazon product reviews and metadata using Python and machine learning techniques. It explores customer sentiment, product performance, recommendations, review classification, and frequently purchased products.

## Project Overview

The notebook uses data from the [Amazon Reviews 2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) and focuses on the following categories:

- All Beauty
- Digital Music
- Video Games
- Handmade Products
- Health and Personal Care

## Main Features

- Data collection and preprocessing
- Missing-value handling and numerical normalization
- Text preprocessing, including:
  - Tokenization
  - Negation handling
  - Emoji and emoticon conversion
  - Contraction expansion
- Exploratory analysis of ratings and products
- Word clouds for low-rated products
- Monthly rating trend analysis
- DistilBERT-based sentiment analysis
- Product feature engineering:
  - Unit price
  - Weighted rating
  - Product quantity extraction
- Product clustering using:
  - TF-IDF features
  - Price
  - Rating
  - K-Means clustering
- Recommendation systems:
  - User-based collaborative filtering
  - Item-based collaborative filtering
  - Word2Vec content-based filtering
  - Hybrid recommendations
- Sentiment classification using:
  - TF-IDF
  - FastText embeddings
  - Naive Bayes
  - K-Nearest Neighbors
  - Random Forest
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Product association rules using the Apriori algorithm

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK
- spaCy
- Hugging Face Transformers
- Gensim
- FastText
- mlxtend
- WordCloud

## Dataset

The project uses the Amazon Reviews 2023 dataset from Hugging Face:

[McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

The notebook downloads a sample of reviews and product metadata for each selected category.

## Requirements

Install the required packages with:

```bash
pip install pandas numpy matplotlib seaborn datasets transformers negspacy emoji nltk gensim contractions spacy wordcloud scikit-learn mlxtend tabulate
