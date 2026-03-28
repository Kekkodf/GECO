import geco.src.helpers as helpers
import torch
import numpy as np
import logging
import vec2text
import transformers
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer

class GECO:
    """
    GECO: GEnerating Contextually Obfuscated Texts
    -------------------------
    *GECO (GEnerating Contextually Obfuscated Texts)* is a text obfuscation mechanism that generates a privatised version of an input text by selecting a pin query from a pool of queries based on the exponential mechanism and then reconstructing a new text from the embedding of the selected query using a corrector model. The mechanism is designed to provide differential privacy guarantees while maintaining the utility of the obfuscated text.

    The main steps of the GECO mechanism are as follows:
    1. Encode the input text into an embedding using a sentence transformer model.
    2. Use the exponential mechanism to select a query from a pool of queries based on a utility function (e.g. cosine similarity)
    3. Reconstruct a new text from the embedding of the selected query using a corrector model.

    The GECO class provides methods for obfuscating a single text or a batch of texts, and it also includes diagnostics such as cosine similarity between the input and output embeddings, and the top candidates from the exponential mechanism.
    """

    def __init__(
        self,
        log: logging.Logger,
        st_model: SentenceTransformer,
        inversion_model: str,
        corrector_model: str,
        utility_function: str = "cosine_similarity",
        top_percentile: float = 0.001,
        cache_path: str = "msmarco_gtr_embeddings.pt",
        **kwargs) -> None:
        """
            :param log: Logger file, if None it is initialised as GECO.Obfuscator.log in the logging directory ./log
            :type log: logging.Logger
            
            :param st_model: the sentence Transformer model to use for embedding the input texts and the pool of queries queries, it is also used as the encoder for the corrector model, so it is important to use a model that is compatible with the corrector. We suggest to use "gtr-t5-base" for better performance of the obfuscation mechanism, as it is the same model trained as the corrector.
            :type st_model: SentenceTransformer

            :param inversion_model: the vec2text inversion model to use for inverting the embedding of the selected query.
            :type inversion_model: vec2text.models.InversionModel

            :param corrector: the vec2text corrector model to use for reconstructing the output text from the embedding of the selected query.
            :type corrector: vec2text.trainers.Corrector

            :param utility_function: The utility function to use for selecting queries. Defaults to "cosine_similarity".
            :type utility_function: str, optional

            :param top_percentile: The percentile of top candidates to consider. Defaults to 0.001.
            :type top_percentile: float, optional

            :param cache_path: The path to the cache file for storing query embeddings. Defaults to "msmarco_gtr_embeddings.pt".
            :type cache_path: str, optional

                :param kwargs: additional keyword arguments.

            :return: None
            :rtype: None
        """
        #initialisation
        if log is None:
            log = mylogger("GECO.Obfuscator")
        self.log = log
        assert st_model is not None, "st_model cannot be None"
        self.st_model = st_model
        assert corrector_model is not None, "corrector_model cannot be None"
        self.corrector_model = corrector_model
        assert inversion_model is not None, "inversion_model cannot be None"
        self.inversion_model = inversion_model
        assert 0.0 < top_percentile <= 1.0, "top_percentile must be in (0, 1]"
        self.top_percentile = top_percentile
        
        self.collection_name = "msmarco-passage/train"
        self.cache_path = cache_path
        self.encode_batch_size = kwargs.get("encode_batch_size", 512)

        self.queries = helpers.load_pool_of_queries(self.log, collection_name=self.collection_name)
        self.pool_of_queries, self.corpus_embs = helpers.build_or_load_pool_of_queries_embeddings(
            self.log, st_model, self.queries, cache_path=self.cache_path,
            encode_batch_size=self.encode_batch_size,
        )
        self.n_corpus = len(self.pool_of_queries)

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

        self.corrector = vec2text.trainers.Corrector(
        model=corrector_model,
        inversion_trainer=inversion_trainer,
        args=None,
        data_collator=vec2text.collator.DataCollatorForCorrection(
            tokenizer=inversion_trainer.model.tokenizer
        ),
    )

    def obfuscate(
        self,
        log: logging.Logger,
        text: str,
        epsilon: float,
        seed: Optional[int] = None,
        top_k_report: int = 5,
    ) -> Tuple[str, str, float, float, float, List[Tuple[str, float, float]]]:
        """
        Obfuscate a single input text using the GECO mechanism.

        :param log: logger to use for logging the obfuscation process.
        :type log: logging.Logger
        :param text: the input text to obfuscate.
        :type text: str
        :param epsilon: the privacy budget for the exponential mechanism.
        :type epsilon: float
        :param seed: random seed for reproducibility. Defaults to None.
        :type seed: Optional[int], optional
        :param top_k_report: the number of top candidates to report from the exponential mechanism for diagnostics. Defaults to 5.
        :type top_k_report: int, optional

        :return: a tuple containing the obfuscated text, the selected query, the cosine similarity between the input and output embeddings, the cosine similarity between the input embedding and the selected query embedding, the angle between the input and output embeddings in degrees, and a list of the top candidates from the exponential mechanism with their probabilities and utilities.
        :rtype: Tuple[str, str, float, float, float, List[Tuple[str, float, float]]]

        Example usage:
        >>> log = create('mylogger')
        >>> st_model, corrector = load_gtr_corrector(log)
        >>> obfuscator = GECO(log, st_model, corrector)
        >>> text = "This is a test sentence to obfuscate."
        >>> epsilon = 1.0
        >>> obfuscated_text, selected_query, cos_io, cos_iq, angle, top_candidates = obfuscator.obfuscate(log, text, epsilon)
        """
        log.info(f"[obfuscate] eps={epsilon:.2f} t={self.top_percentile:.2f} "
                 f"text='{text[:60]}'")

        # encoding input text
        e = helpers.embed(self.log, self.st_model, [text])[0]

        # exponential mechanism to select the pin
        selected_query, selected_emb, probs, utilities = exponential_mechanism(
            self.log, e, self.corpus_embs, self.corpus_queries, epsilon,
            top_percentile=self.top_percentile, delta_u=2.0, seed=seed,
        )

        # Generation of the output text
        log.info(f"[reconstruct pivot] q*: '{selected_query[:60]}'")
        output_text = generate_from_embedding(selected_emb, self.corrector)
        log.info(f"[reconstruct output] '{output_text}'")

        # Diagnostics
        out_emb = helpers.embed(self.log, self.st_model, [output_text])[0]
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

        return (output_text, selected_query, cos_io, cos_iq, angle, top_candidates
        )

    def batch_obfuscate(self, texts: List[str], epsilon: float,
                        seed: Optional[int] = None) -> List[Tuple[str, str, float, float, float, List[Tuple[str, float, float]]]]:
        return [self.obfuscate(t, epsilon, seed=seed) for t in texts]
    

@staticmethod
def exponential_mechanism(
        log: torch.Tensor,
        e: torch.Tensor,
        pool_embs: torch.Tensor,
        pool_queries: List[str],
        epsilon: float,
        top_percentile: float = 0.1,
        utility_function: str = "cosine_similarity",
        seed: Optional[int] = None,
        **kwargs) -> Tuple[str, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Implemenmts the exponential mechanism for selecting a query from the queries pool

    
    Args:
        log (logging.Logger): logger to use for logging the process.
        e (torch.Tensor): the embedding of the input text, shape (d,).
        pool_embs (torch.Tensor): the embeddings of the queries pool, shape (N, d).
        pool_queries (List[str]): the list of queries in the pool, length N.
        epsilon (float): the privacy budget for the exponential mechanism.
        top_percentile (float, optional): the percentile threshold for filtering candidates based on utility. Defaults to 0.1 (i.e., top 10%).
        utility_function (str, optional): the utility function to use. Defaults to "cosine_similarity".
        seed (Optional[int], optional): random seed for reproducibility. Defaults to None.

    Returns:
        Tuple[str,torch.Tensor,np.ndarray,np.ndarray]: A tuple containing the selected query, its embedding, the true EM probabilities, and the utilities.

        selected_query: the string query selected by the exponential mechanism from the query pool. It is the pin from which the reconstruction will be performed.
        selected_embedding: the embedding of the selected query, shape (d,).
        probs: the true probabilities of selection for each candidate in the filtered pool, shape (K_filtered,). K_filtered is the number of candidates that passed the percentile filter. These probabilities are not used for sampling (the Gumbel-max trick is used instead), but are returned for diagnostics.
        utilities: the utility values (cosine similarities) for each candidate in the filtered pool, shape (K_filtered,). These are the values used in the exponential mechanism to determine selection probabilities.
    """
    #sanity checks on percentiles and utility function
    if not (0.0 < top_percentile <= 1.0):
        raise ValueError(f"top_percentile must be in (0, 1], got {top_percentile}") #TODO: add this check to the obfuscator init as well
    if seed is not None:
        np.random.seed(seed) #TODO: Set it also in the init
    if utility_function not in ["cosine_similarity", "rbf", "correlation"]:
        raise ValueError(f"Unsupported utility function: {utility_function}. Supported functions are: 'cosine_similarity', 'rbf', 'correlation'.")
    if utility_function == "cosine_similarity":
        delta_u = 2.0 # L2 sensitivity of cosine similarity with normalized embeddings (max change in cosine similarity when one point changes is 2, cosine similarity ranges from -1 to 1, and changing one point can flip the similarity from 1 to -1 or vice versa)
    elif utility_function == "rbf":
        delta_u = 1.0 # L2 sensitivity of RBF kernel with normalized embeddings and gamma=1 (max change in RBF similarity when one point changes is 1, RBF similarity ranges from 0 to 1, and changing one point can flip the similarity from 1 to 0 or vice versa) -- NOTE: this is a very loose upper bound, the actual sensitivity of the RBF kernel can be much smaller depending on the gamma parameter and the distribution of the embeddings, but using a smaller delta_u would require a more complex analysis and we prefer to use a conservative upper bound for simplicity.
    elif utility_function == "correlation":
        delta_u = 2.0 # L2 sensitivity of correlation with normalized embeddings (max change in correlation when one point changes is 2, correlation ranges from -1 to 1, and changing one point can flip the correlation from 1 to -1 or vice versa)

    e_cpu = (e / e.norm()).cpu()
    N = len(pool_queries)

    # Compute utilities in chunks to bound peak RAM
    chunk = 50_000
    all_utils = np.empty(N, dtype=np.float32)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        all_utils[start:end] = (pool_embs[start:end] @ e_cpu).numpy()

    # Percentile filter
    cutoff = float(np.percentile(all_utils, 100.0 * (1.0 - top_percentile)))
    mask = all_utils >= cutoff

    # Get indices of candidates that passed the filter
    filtered_idx = np.where(mask)[0]
    utilities = all_utils[filtered_idx]
    K = len(filtered_idx)
    log.info(
        f"[Exp. Mech.] t = {top_percentile:.2f} => {K:,}/{N:,} candidates "
        f"(u in [{utilities.min():.4f}, {utilities.max():.4f}])"
    )

    # Gumbel-max trick (equivalent to sampling from Exp Mech distribution) ref. https://differentialprivacy.org/one-shot-top-k/#:~:text=Gumbel%20Noise%20and%20the%20Exponential,%CE%B2%20)%20=%201%20%CE%B2%20exp%20%E2%81%A1
    gumbel_scale = (2.0 * delta_u) / epsilon
    noisy = utilities + np.random.gumbel(0.0, gumbel_scale, size=K)
    local_best = int(np.argmax(noisy))
    global_best = int(filtered_idx[local_best])

    # True Exp Mech probabilities for diagnostics (not used in sampling)
    log_w = (epsilon / (2.0 * delta_u)) * utilities
    log_w -= log_w.max()
    probs = np.exp(log_w); probs /= probs.sum()

    return pool_queries[global_best], pool_embs[global_best], probs, utilities

@staticmethod
def generate_from_embedding(
    target_emb: torch.Tensor,
    corrector: vec2text.trainers.Corrector,
    **kwargs) -> str:
    """
    Generate a text from the target embedding using the corrector model.

    Args:
        target_emb (torch.Tensor): the target embedding from which to generate the text, shape (d,).
        corrector (vec2text.trainers.Corrector): the vec2text corrector model to use for generation.
        kwargs: additional keyword arguments, such as num_corrector_steps, which determines how many steps of the corrector to run for generation. By default, it is set to 1, which means that the corrector will be applied once to the target embedding to generate the output text. Increasing this number may lead to better reconstruction quality but also increases the computational cost and weakens privacy.

    Returns:
        str: the generated text from the target embedding.
    """
    device = next(corrector.model.parameters()).device
    with torch.no_grad():
        texts = vec2text.invert_embeddings(
            embeddings=target_emb.unsqueeze(0).to(device),
            corrector=corrector,
            num_steps=kwargs.get("num_corrector_steps", 1),
        )
    return texts[0].strip()