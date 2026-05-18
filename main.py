import argparse
import os
import pandas as pd
import ir_datasets

from geco.utils import createLogger
from geco.src.AbstractGECO import Obfuscator
from geco.src.CorrelationGECO import CorrelationGECO
from geco.src.CosineGECO import CosineGECO
from geco.src.RbfGECO import KernelDensityGECO

collections = [
    "msmarco-passage/trec-dl-2019/judged",
    "msmarco-passage/trec-dl-2020/judged",
    "disks45/nocr/trec-robust-2004",
    "medline/2004/trec-genomics-2004",
]

col2name = {
    "medline/2004/trec-genomics-2004": "Med04",
    "msmarco-passage/trec-dl-2019/judged": "DL19",
    "msmarco-passage/trec-dl-2020/judged": "DL20",
    "disks45/nocr/trec-robust-2004": "Rob04",
}



EPSILONS = [1, 5, 10, 15, 20, 50]
N_RUNS = 10
OUTPUT_DIR = "results"


def main():
    log = createLogger('main')

    # Map a friendly name -> obfuscator class so we can build them in a loop
    geco_classes = {
        "CosineGECO": CosineGECO,
        "KernelDensityGECO": KernelDensityGECO,
        "CorrelationGECO": CorrelationGECO,
    }

    for col in collections:
        log.info(f"Loading collection {col}")
        dataset = ir_datasets.load(col)
        queries_df = pd.DataFrame(dataset.queries_iter())
        if col == "medline/2004/trec-genomics-2004" or col == "disks45/nocr/trec-robust-2004":
            # keep only the columns 'query_id' and 'title'
            queries_df = queries_df[['query_id', 'title']]
            # rename the columns to 'query_id' and 'text'
            queries_df.columns = ['query_id', 'text']
        
        for geco_name, GecoClass in geco_classes.items():
            log.info(f"Initializing {geco_name}")
            geco_obfuscator = GecoClass(log=log)
            print(geco_obfuscator)

            for epsilon in EPSILONS:
                print("\n" + "=" * 76)
                print(f"Running {geco_name} with epsilon={epsilon}")

                # One row per (query, run) pair — long format
                rows = []
                for _, q in queries_df.iterrows():
                    query_id = q['query_id']
                    text = q['text']
                    for _ in range(N_RUNS):
                        obfuscated, debug_info = geco_obfuscator.obfuscate(text, epsilon)

                        rows.append({
                            "query_id": query_id,
                            "text": text,
                            "obfuscatedText": obfuscated,
                            "mechanism": geco_name,
                            "epsilon": epsilon,
                            "top_percentile": geco_obfuscator.top_percentile,
                            "pivot_query": debug_info[0],
                            "pool": geco_obfuscator.collection_name,
                            "corrector_model": geco_obfuscator.corrector_model_name,
                        })

                out_df = pd.DataFrame(
                    rows,
                    columns=["query_id", "text", "obfuscatedText", "mechanism", "epsilon", "top_percentile", "pivot_query", "pool", "corrector_model"],
                )

                os.makedirs(OUTPUT_DIR+f'/{geco_name}/{col2name[col]}', exist_ok=True)
                out_path = os.path.join(
                    OUTPUT_DIR,
                    f"obfuscatedText_{geco_name}_{epsilon}.csv",
                )
                out_df.to_csv(out_path, index=False)
                log.info(f"Saved {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()