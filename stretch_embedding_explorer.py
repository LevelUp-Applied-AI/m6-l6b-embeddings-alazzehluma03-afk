import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import torch

from embeddings_lab import load_glove, extract_bert_embedding


# 1. Word Categories (~200 words)

categories = {
    "countries": [
        "france","germany","italy","spain","china","japan","india","brazil",
        "canada","mexico","russia","egypt","turkey","iran","iraq","syria",
        "saudi","uae","qatar","kuwait","argentina","chile","peru","colombia",
        "australia","newzealand","sweden","norway","denmark","finland",
        "poland","ukraine","greece","portugal","netherlands","belgium",
        "switzerland","austria","ireland","scotland"
    ],
    "sports": [
        "football","soccer","tennis","cricket","basketball","goal","team","match",
        "player","coach","league","score","win","loss","draw","tournament",
        "champion","cup","final","season","referee","stadium","fans","club",
        "striker","defender","midfielder","goalkeeper","penalty","injury",
        "training","fitness","competition","medal","olympics","race","track",
        "field","athlete","record"
    ],
    "business": [
        "market","stock","trade","bank","money","economy","investment","profit",
        "loss","revenue","finance","capital","shares","growth","inflation",
        "interest","debt","credit","fund","budget","tax","income","spending",
        "cost","price","demand","supply","export","import","industry","company",
        "corporate","sales","earnings","dividend","assets","liability",
        "exchange","currency","risk"
    ],
    "tech": [
        "computer","software","internet","ai","data","network","system","digital",
        "hardware","program","code","algorithm","machine","learning","cloud",
        "server","database","security","encryption","cyber","robot","automation",
        "device","mobile","app","platform","web","online","technology","innovation",
        "startup","engineering","development","interface","protocol","signal",
        "processor","memory","storage","virtual"
    ],
    "emotions": [
        "happy","sad","angry","love","fear","joy","excited","stress",
        "anxiety","hope","trust","surprise","disgust","calm","proud",
        "shame","guilt","relief","pain","pleasure","hate","envy","jealous",
        "lonely","bored","confused","worried","nervous","grateful","kind",
        "friendly","optimistic","pessimistic","motivated","tired","relaxed",
        "curious","serious","funny","emotional"
    ]
}


# 2. Build word embeddings


def build_word_matrix(categories, glove):
    words, labels, vectors = [], [], []

    for category, word_list in categories.items():
        for w in word_list:
            if w in glove:
                words.append(w)
                labels.append(category)
                vectors.append(glove[w])

    return words, labels, np.array(vectors)


# 3. PCA reduction

def reduce_pca(vectors):
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(vectors)


# 4. Plot words

def plot_words(points, words, labels):
    plt.figure(figsize=(10, 8))

    colors = {
        "countries": "orange",
        "sports": "purple",
        "business": "green",
        "tech": "blue",
        "emotions": "red"
    }

    for lab in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == lab]
        plt.scatter(
            points[idxs, 0],
            points[idxs, 1],
            label=lab,
            s=50,
            alpha=0.7,
            color=colors.get(lab, None)
        )

    important_words = ["china", "germany", "football", "bank", "computer", "happy"]

    for i, w in enumerate(words):
        if w in important_words:
            plt.annotate(w, (points[i, 0], points[i, 1]))

    plt.legend()
    plt.title("GloVe Word Embeddings (PCA)")
    plt.tight_layout()
    plt.savefig("word_plot.png")
    plt.show()


# 5. Document embeddings


def get_sample_articles(df):
    samples = []
    for cat in df["category"].unique():
        samples.extend(df[df["category"] == cat].head(4).to_dict("records"))
    return samples[:20]


def build_doc_embeddings(texts, tokenizer, model):
    return np.array([
        extract_bert_embedding(t, tokenizer, model)
        for t in texts
    ])


# 6. Plot documents

def plot_docs(points, texts, labels):
    plt.figure(figsize=(10, 8))

    for lab in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == lab]
        plt.scatter(points[idxs, 0], points[idxs, 1], label=lab, s=60, alpha=0.7)

    for i in range(len(texts)):
        plt.annotate(f"{labels[i][:3]}-{i}", (points[i, 0], points[i, 1]))

    plt.legend()
    plt.title("BBC Articles (DistilBERT + PCA)")
    plt.tight_layout()
    plt.savefig("doc_plot.png")
    plt.show()


# MAIN

if __name__ == "__main__":
    # GloVe
    glove = load_glove("data/glove_50k_50d.txt")
    words, labels, vectors = build_word_matrix(categories, glove)

    word_points = reduce_pca(vectors)
    plot_words(word_points, words, labels)

    # BERT
    df = pd.read_csv("data/bbc_news.csv")
    samples = get_sample_articles(df)

    texts = [s["text"] for s in samples]
    cats = [s["category"] for s in samples]

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    doc_vectors = build_doc_embeddings(texts, tokenizer, model)
    doc_points = reduce_pca(doc_vectors)

    plot_docs(doc_points, texts, cats)