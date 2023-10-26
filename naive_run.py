import argparse
import itertools
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from scholarly import scholarly

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    # parser.add_argument("criteria", type=str)
    parser.add_argument("--max_authors", type=int, default=5)
    args = parser.parse_args()

    google_scholar_dir = Path("data/google_scholar/")
    cat_path = google_scholar_dir / "categories.json"
    sub_cat_path = google_scholar_dir / "sub_categories.json"

    with cat_path.open() as cat_f:
        categories = json.load(cat_f)
    with sub_cat_path.open() as sub_cat_f:
        sub_categories = json.load(sub_cat_f)

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = st_model.encode([args.query], convert_to_tensor=True)
    assert isinstance(query_embedding, torch.Tensor)
    sub_cat_embeddings = torch.load(google_scholar_dir / "sub_cat_embeddings.pt")

    cat_searches = semantic_search(query_embedding, sub_cat_embeddings, top_k=2)[0]
    found_cats = [sub_categories[match["corpus_id"]] for match in cat_searches]
    print("Most relevant topics are:")
    for cat, match in zip(found_cats, cat_searches):
        print(cat, match["score"])

    authors = list(itertools.islice(scholarly.search_keywords(found_cats), args.max_authors))
    if len(authors) < 0:
        print("Found nobody")

    print("Authors found for both topics:")
    for author in authors:
        print("---")
        print("Name:", author["name"])
        print("Affiliation:", author["affiliation"])
        print("Email:", author["email_domain"])
        print("Google Scholar ID:", author["scholar_id"])
        print("Interests:", author["interests"])

