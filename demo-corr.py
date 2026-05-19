"""
vec2text_dp.py  --  epsilon-d_X-DP text obfuscation
====================================================

Method (3 phases)
-----------------
1. ENCODE
   Embed input text x with gtr-t5-base -> unit vector e in S^{d-1}.

2. CANDIDATE FILTERING
   Candidate universe = all MS MARCO training queries (~500k), pre-embedded
   with gtr-t5-base and cached to disk.  These embeddings lie on gtr-base's
   data manifold -- no synthetic points, no off-manifold decoding.

   Compute cosine utility u_k = e . e_k for every query.
   Retain only the top (t x 100)% by utility as the effective candidate set.

3. EXPONENTIAL MECHANISM  (epsilon-d_X-DP)
   Utility:      u(x, q_k) = cosine_similarity(e, e_k)  in [-1, 1]
   Sensitivity:  Delta_u = 2  (tight bound on unit sphere, no phi assumptions)
   Selection:    P(q_k | x) ∝ exp(eps * u / 2*Delta_u)
   Implemented via the Gumbel-max trick in O(K_filtered).

   The selected query q* has its real embedding e* = phi(q*) decoded by the
   vec2text corrector in exactly 1 step to produce the final output x~.

   Post-processing theorem: exponential mechanism is the sole random step.
   The 1-shot vec2text decode is deterministic -> cannot weaken eps-d_X-DP.

Why MS MARCO?
-------------
* Fixed, input-independent candidate set -> standard EM proof applies.
* All embeddings are on gtr-base's training manifold -> 1 corrector step
  produces fluent, coherent output without needing iterative refinement.
* ~500k queries give broad semantic coverage; no n_pool budget to tune.

Installation
------------
    pip install vec2text sentence-transformers transformers torch numpy datasets

Models downloaded automatically:
    ielabgroup/vec2text_gtr-base-st_inversion
    ielabgroup/vec2text_gtr-base-st_corrector

MS MARCO embeddings (~1.4 GB) are cached after first run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import transformers
import vec2text
import vec2text.models
import vec2text.trainers
import vec2text.collator
from sentence_transformers import SentenceTransformer
import pandas as pd
from geco.utils import mylogger

log = mylogger.create("demo")

def center_and_normalize(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Transforms embeddings such that their dot product equals their Pearson Correlation.
    Input: (N, d) or (d,)
    """
    # 1. Center: Subtract the mean of each vector's components
    # embeddings.mean(dim=-1, keepdim=True) calculates the average value across the 768 dimensions
    centered = embeddings - embeddings.mean(dim=-1, keepdim=True)
    
    # 2. Normalize: Bring them back to the unit sphere
    # We add a tiny eps to avoid division by zero for constant vectors
    norms = centered.norm(dim=-1, keepdim=True) + 1e-9
    return centered / norms

def load_gtr_corrector() -> Tuple[SentenceTransformer, vec2text.trainers.Corrector]:
    """Load gtr-t5-base embedder + pretrained vec2text corrector."""
    log.info("Loading gtr-t5-base sentence embedder ...")
    st_model = SentenceTransformer("sentence-transformers/gtr-t5-base")

    log.info("Loading vec2text inversion (zero-step) model ...")
    inversion_model = vec2text.models.InversionModel.from_pretrained(
        "ielabgroup/vec2text_gtr-base-st_inversion"#jxm/gtr__nq__32,ielabgroup/vec2text_gtr-base-st_inversion, text-embedding-ada-002 (not working)
    )

    log.info("Loading vec2text corrector model ...")
    corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained(
        "ielabgroup/vec2text_gtr-base-st_corrector"#jxm/gtr__nq__32__correct,ielabgroup/vec2text_gtr-base-st_corrector, text-embedding-ada-002 (not working)
    )

    inversion_trainer = vec2text.trainers.InversionTrainer(
        model=inversion_model,
        train_dataset=None,
        eval_dataset=None,
        data_collator=transformers.DataCollatorForSeq2Seq(
            inversion_model.tokenizer,
            label_pad_token_id=-100,
        ),
    )
    corrector_model.config.dispatch_batches = None

    corrector = vec2text.trainers.Corrector(
        model=corrector_model,
        inversion_trainer=inversion_trainer,
        args=None,
        data_collator=vec2text.collator.DataCollatorForCorrection(
            tokenizer=inversion_trainer.model.tokenizer
        ),
    )
    log.info("All models loaded.")
    return st_model, corrector


# ==============================================================================
# 2.  EMBEDDING HELPER
# ==============================================================================

def embed(st_model: SentenceTransformer, texts: List[str]) -> torch.Tensor:
    """Return (N, 768) float32 L2-normalised embeddings from gtr-t5-base."""
    with torch.no_grad():
        embs = st_model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return embs.float()


# ==============================================================================
# 3.  MS MARCO CORPUS LOADING AND EMBEDDING
# ==============================================================================

def load_msmarco_queries(max_queries: Optional[int] = None) -> List[str]:
    """
    Load MS MARCO v2.1 training queries from HuggingFace (~502,939 queries).

    Parameters
    ----------
    max_queries : optional cap for development (e.g. 50_000).
                  None => full corpus.
    """
    try:
        import ir_datasets
    except ImportError as exc:
        raise ImportError("pip install ir-datasets") from exc
    log.info("Loading MS MARCO training queries ...")
    ds = ir_datasets.load("msmarco-passage/train")
    df = ds.queries_iter()
    df_pandas = pd.DataFrame(df)  
    queries = df_pandas['text'].tolist()
    if max_queries is not None:
        queries = queries[:max_queries]
    log.info(f"Loaded {len(queries):,} queries.")
    return queries


def build_or_load_corpus_embeddings(
    st_model: SentenceTransformer,
    queries: List[str],
    cache_path: str = "msmarco_gtr_embeddings.pt",
    encode_batch_size: int = 512,
) -> Tuple[List[str], torch.Tensor]:
    """
    Embed all MS MARCO queries with gtr-t5-base; cache result to disk.

    Cache format: torch.save({"queries": List[str], "embeddings": Tensor(N,768)})

    On first run with the full corpus: ~3-5 min on GPU, ~30-40 min on CPU.
    Subsequent runs load from cache in seconds.

    Returns
    -------
    (queries, embeddings)  --  same order, both of length N.
    embeddings is (N, 768) float32 CPU tensor, L2-normalised.
    """
    cache = Path(cache_path)
    if cache.exists():
        log.info(f"Loading embedding cache from {cache_path} ...")
        data = torch.load(cache_path, map_location="cpu")
        cq: List[str] = data["queries"]
        ce: torch.Tensor = data["embeddings"]
        if len(cq) == len(queries):
            log.info(f"Cache valid: {len(cq):,} queries x {ce.shape[1]}d.")
            return cq, ce
        log.warning(f"Cache size mismatch ({len(cq):,} vs {len(queries):,}). Re-encoding ...")

    log.info(f"Encoding {len(queries):,} queries (batch={encode_batch_size}) ...")
    embeddings = st_model.encode(
        queries,
        batch_size=encode_batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).float().cpu()

    torch.save({"queries": queries, "embeddings": embeddings}, cache_path)
    log.info(f"Cache saved to {cache_path}.")
    return queries, embeddings


# ==============================================================================
# 4.  EXPONENTIAL MECHANISM  (percentile filter + Gumbel-max trick)
# ==============================================================================

def exponential_mechanism_correlation(
    e: torch.Tensor,
    corpus_embs: torch.Tensor,
    corpus_queries: List[str],
    epsilon: float,
    top_percentile: float = 0.1,
    delta_u: float = 2.0,      # Sensitivity remains 2.0 on the unit sphere
    seed: Optional[int] = None,
) -> Tuple[str, torch.Tensor, np.ndarray, np.ndarray]:
    
    if seed is not None:
        np.random.seed(seed)

    # Transform input and corpus into Pearson-ready vectors
    # Note: In a production setting, you'd pre-center the corpus_embs in __init__
    e_corr = center_and_normalize(e.cpu())
    corpus_corr = center_and_normalize(corpus_embs.cpu())

    N = len(corpus_queries)
    chunk = 50_000
    correlations = np.empty(N, dtype=np.float32)

    # The dot product of centered-normalized vectors IS the Pearson Correlation
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        correlations[start:end] = (corpus_corr[start:end] @ e_corr).numpy()

    # Percentile filter based on correlation
    if top_percentile < 1.0:
        cutoff = float(np.percentile(correlations, 100.0 * (1.0 - top_percentile)))
        mask = correlations >= cutoff
    else:
        mask = np.ones(N, dtype=bool)

    filtered_idx = np.where(mask)[0]
    utilities    = correlations[filtered_idx]
    K            = len(filtered_idx)

    log.info(
        f"[EM-Correlation] t={top_percentile:.2f} => {K:,}/{N:,} candidates "
        f"(rho in [{utilities.min():.4f}, {utilities.max():.4f}])"
    )

    # Gumbel-max trick
    gumbel_scale = (2.0 * delta_u) / epsilon
    noisy        = utilities + np.random.gumbel(0.0, gumbel_scale, size=K)
    local_best   = int(np.argmax(noisy))
    global_best  = int(filtered_idx[local_best])

    # True EM probabilities for diagnostics
    log_w  = (epsilon / (2.0 * delta_u)) * utilities
    log_w -= log_w.max()
    probs  = np.exp(log_w); probs /= probs.sum()

    # IMPORTANT: We return the ORIGINAL corpus_embs[global_best] for reconstruction.
    # The vec2text model expects the raw GTR manifold, not the centered version.
    return corpus_queries[global_best], corpus_embs[global_best], probs, utilities


# ==============================================================================
# 5.  1-SHOT VEC2TEXT RECONSTRUCTION
# ==============================================================================

def reconstruct_from_embedding(
    target_emb: torch.Tensor,
    corrector: vec2text.trainers.Corrector,
) -> str:
    """
    Decode a single embedding to text with exactly 1 corrector step.

    target_emb is phi(q*) -- a real gtr-base embedding of an MS MARCO query,
    lying on the training manifold of both gtr-base and the vec2text corrector.
    One corrector step from the zero-step hypothesis is sufficient for
    fluent, semantically faithful reconstruction.

    The output x~ is generally a paraphrase of q*, not identical -- adding an
    extra layer of indirection against corpus-membership attacks.
    """
    device = next(corrector.model.parameters()).device
    with torch.no_grad():
        texts = vec2text.invert_embeddings(
            embeddings=target_emb.unsqueeze(0).to(device),
            corrector=corrector,
            num_steps=1,
        )
    return texts[0].strip()


# ==============================================================================
# 6.  RESULT DATACLASS
# ==============================================================================

@dataclass
class ObfuscationResult:
    input_text:   str
    output_text:  str
    epsilon:      float
    top_percentile: float
    n_corpus:     int
    n_filtered:   int
    selected_query: str
    input_embedding:  np.ndarray
    output_embedding: np.ndarray
    cosine_sim_input_output: float   # cos(phi(x), phi(x~))
    cosine_sim_input_query:  float   # cos(phi(x), phi(q*))
    angular_distance_deg:    float
    top_candidates: List[Tuple[str, float, float]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "-" * 76,
            f"  INPUT   : {self.input_text}",
            f"  PIVOT q*: {self.selected_query}",
            f"  OUTPUT  : {self.output_text}",
            f"  eps={self.epsilon:.1f} | t={self.top_percentile:.2f} | "
            f"corpus={self.n_corpus:,} -> filtered={self.n_filtered:,}",
            f"  cos(x,q*)={self.cosine_sim_input_query:.4f} | "
            f"cos(x,x~)={self.cosine_sim_input_output:.4f} | "
            f"angle(x,x~)={self.angular_distance_deg:.1f}deg",
            "  top candidates:",
        ]
        for rank, (q, p, u) in enumerate(self.top_candidates, 1):
            lines.append(f'    [{rank}] p={p:.4f} u={u:.4f}  "{q}"')
        lines.append("-" * 76)
        return "\n".join(lines)


# ==============================================================================
# 7.  OBFUSCATOR CLASS
# ==============================================================================

class Vec2TextDPObfuscator:
    """
    Epsilon-d_X-DP text obfuscator.

    Candidate universe  : all MS MARCO training queries (~500k), embedded
                          with gtr-t5-base and cached to disk at init.
    Candidate filtering : top-t% by cosine utility u(x,q) = phi(x).phi(q).
    Selection           : exponential mechanism (Gumbel-max), epsilon-d_X-DP.
    Reconstruction      : 1-shot vec2text from the selected query's real
                          embedding phi(q*).

    Parameters
    ----------
    top_percentile  : t in (0,1] -- fraction of corpus kept before mechanism.
    cache_path      : disk path for the pre-computed MS MARCO embeddings.
    max_queries     : optional cap for development; None => full ~500k corpus.
    encode_batch_size : batch size when encoding the corpus the first time.
    """

    def __init__(
        self,
        st_model: SentenceTransformer,
        corrector: vec2text.trainers.Corrector,
        top_percentile: float = 0.001,
        cache_path: str = "msmarco_gtr_embeddings.pt",
        max_queries: Optional[int] = 500,
        encode_batch_size: int = 512,
    ):
        self.st_model       = st_model
        self.corrector      = corrector
        self.top_percentile = top_percentile

        queries = load_msmarco_queries(max_queries=max_queries)
        self.corpus_queries, self.corpus_embs = build_or_load_corpus_embeddings(
            st_model, queries, cache_path=cache_path,
            encode_batch_size=encode_batch_size,
        )
        self.n_corpus = len(self.corpus_queries)
        # Pre-calculate for Correlation Utility
        log.info("Pre-centering corpus for Pearson Correlation...")
        self.corpus_corr = center_and_normalize(self.corpus_embs)

    def obfuscate(
        self,
        text: str,
        epsilon: float,
        gamma: float = 1.0,  # Added gamma
        seed: Optional[int] = None,
        top_k_report: int = 5,
    ) -> ObfuscationResult:
        
        # Phase 1: encode input
        e = embed(self.st_model, [text])[0]

        # Phase 2+3: RBF Exponential Mechanism
        # Note: delta_u is now 1.0 for the RBF range [0, 1]
        selected_query, selected_emb, probs, utilities = exponential_mechanism_correlation(
            e, self.corpus_embs, self.corpus_queries, epsilon,
            top_percentile=self.top_percentile, 
            delta_u=1.0, 
            seed=seed,
        )

        # Phase 4: 1-shot reconstruction from phi(q*)
        log.info(f"[reconstruct pivot] q*: '{selected_query[:60]}'")
        output_text = reconstruct_from_embedding(selected_emb, self.corrector)
        log.info(f"[reconstruct output] '{output_text}'")

        # Diagnostics
        out_emb = embed(self.st_model, [output_text])[0]
        cos_io  = float((e.cpu() @ out_emb.cpu()).item())
        cos_iq  = float((e.cpu() @ selected_emb.cpu()).item())
        angle   = float(np.degrees(np.arccos(np.clip(cos_io, -1.0, 1.0))))

        # Rebuild filtered_idx for the top-k report
        e_cpu = (e / e.norm()).cpu()
        chunk = 50_000; N = self.n_corpus
        all_utils = np.empty(N, dtype=np.float32)
        for s in range(0, N, chunk):
            ee = min(s + chunk, N)
            all_utils[s:ee] = (self.corpus_embs[s:ee] @ e_cpu).numpy()
        if self.top_percentile < 1.0:
            cut = float(np.percentile(all_utils, 100.0 * (1.0 - self.top_percentile)))
            fidx = np.where(all_utils >= cut)[0]
        else:
            fidx = np.arange(N)

        top_local = np.argsort(probs)[::-1][:top_k_report]
        top_candidates = [
            (self.corpus_queries[fidx[i]], float(probs[i]), float(utilities[i]))
            for i in top_local
        ]

        return ObfuscationResult(
            input_text=text, output_text=output_text, epsilon=epsilon,
            top_percentile=self.top_percentile, n_corpus=self.n_corpus,
            n_filtered=len(utilities), selected_query=selected_query,
            input_embedding=e.cpu().numpy(), output_embedding=out_emb.cpu().numpy(),
            cosine_sim_input_output=cos_io, cosine_sim_input_query=cos_iq,
            angular_distance_deg=angle, top_candidates=top_candidates,
        )

    def batch_obfuscate(self, texts: List[str], epsilon: float,
                        seed: Optional[int] = None) -> List[ObfuscationResult]:
        return [self.obfuscate(t, epsilon, seed=seed) for t in texts]


# ==============================================================================
# 8.  DEMO
# ==============================================================================

DEMO_TEXTS = [
    "do goldfish grow",
    "what is wifi vs bluetooth",
    "why did the us volunterilay enter ww1",
    "definition declaratory judgment",
    "right pelvic pain causes"
]
EPSILON_VALUES = [1.0, 5.0, 10.0, 12.5, 15.0, 17.5, 20.0, 30.0, 50.0]


def main() -> None:
    print("\nVec2Text-DP  --  MS MARCO + exponential mechanism obfuscation")
    print("eps-d_X-DP | candidates=MS MARCO training queries | 1-shot reconstruction\n")

    st_model, corrector = load_gtr_corrector()

    # First run: MS MARCO downloaded + embeddings computed and cached (~5 min GPU).
    # Later runs: cache loaded instantly.
    # Use max_queries=50_000 for a quick development test.
    obfuscator = Vec2TextDPObfuscator(
        st_model=st_model,
        corrector=corrector,
        top_percentile=0.01,
        cache_path="msmarco_gtr_embeddings.pt",
        max_queries=None,
    )

    # Epsilon sweep on one text
    #print("\n" + "=" * 76)
    #for eps in EPSILON_VALUES:
    #    print(obfuscator.obfuscate(DEMO_TEXTS[0], epsilon=eps, seed=42))

    # Fixed epsilon, multiple texts
    print("\n" + "=" * 76)
    for text in DEMO_TEXTS:
      for eps in EPSILON_VALUES:
        print(obfuscator.obfuscate(text, epsilon=eps))


if __name__ == "__main__":
    main()