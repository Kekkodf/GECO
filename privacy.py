import os
from sentence_transformers import SentenceTransformer
import pandas as pd

evaluator = SentenceTransformer('all-MiniLM-L6-v2')

collections = ['DL19', 'DL20', 'Med04']
epsilons = [1, 5, 10, 15, 20, 50]
mechanisms = {
    "CosineGECO":        "GECO_cos",
    "CorrelationGECO":   "GECO_cor",
    "KernelDensityGECO": "GECO_rbf",
}

def jaccard(a: str, b: str) -> float:
    sa = set(str(a).lower().split())
    sb = set(str(b).lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)

summary = []

for col in collections:
    cache_path = f"results/privacy/{col}/privacyEvaluation.csv"

    if os.path.exists(cache_path):
        print(f"[{col}] loading cached results from {cache_path}")
        col_df = pd.read_csv(cache_path)
    else:
        print(f"[{col}] computing results")
        col_rows = []
        for eps in epsilons:
            for mech_name, mech_dir in mechanisms.items():
                df = pd.read_csv(f"results/{mech_name}/{col}/obfuscatedText_{mech_name}_{eps}.csv")

                emb_o = evaluator.encode(df['text'].tolist(), convert_to_tensor=True, show_progress_bar=False)
                emb_b = evaluator.encode(df['obfuscatedText'].tolist(), convert_to_tensor=True, show_progress_bar=False)

                df['cosine_similarity']  = evaluator.similarity_pairwise(emb_o, emb_b).cpu().numpy()
                df['jaccard_similarity'] = [jaccard(t, o) for t, o in zip(df['text'], df['obfuscatedText'])]

                per_query = (
                    df.groupby('query_id', as_index=False)[['cosine_similarity', 'jaccard_similarity']]
                      .mean()
                )
                per_query['collection'] = col
                per_query['epsilon']    = eps
                per_query['mechanism']  = mech_name
                col_rows.append(per_query)

        col_df = pd.concat(col_rows, ignore_index=True)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        col_df.to_csv(cache_path, index=False)
        print(f"[{col}] saved cache to {cache_path}")

    summary.append(col_df)

summary_df = pd.concat(summary, ignore_index=True)

pivot = summary_df.pivot_table(
    index=['collection', 'mechanism'],
    columns='epsilon',
    values=['cosine_similarity', 'jaccard_similarity'],
)
print(pivot.to_latex(float_format="%.3f"))