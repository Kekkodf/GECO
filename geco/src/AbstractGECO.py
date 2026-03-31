import geco.src.helpers as helpers
import torch
import numpy as np
import pandas as pd
import logging
import vec2text
import transformers

from pathlib import Path
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer

torch._logging.set_logs(dynamo=logging.WARNING)

class Obfuscator:
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
        st_model_name: str,
        inversion_model_name: str,
        corrector_model_name: str,
        cache_path_pins: str,
        collection_name: str = "msmarco-passage/train",
        epsilons: List[float] = [1.0, 5.0, 10.0, 12.5, 15.0, 17.5, 20.0, 30.0, 50.0],
        **kwargs) -> None:
        """
        Initialisation of the GECO obfuscator Abstract class. 
        
        WARNING: this is an abstract class and should not be instantiated directly. It is meant to be subclassed by specific implementations of the GECO mechanism that may use different models or configurations.

            :param log: logger to use for logging the initialization process.
            :type log: logging.Logger
            :param st_model_name: the name or path of the sentence transformer model to use for embedding the texts. This model will be used to encode both the input texts and the queries in the pool into embeddings. The embeddings will be used in the exponential mechanism to select a query and in the corrector model to reconstruct the obfuscated text.
            :type st_model_name: str
            :param inversion_model_name: the name or path of the vec2text inversion model to use for reconstructing the obfuscated text from the selected query embedding. This model will be used in the generation step of the GECO mechanism to generate a new text from the embedding of the selected query.
            :type inversion_model_name: str
            :param corrector_model_name: the name or path of the vec2text corrector model to use for reconstructing the obfuscated text from the selected query embedding. This model will be used in the generation step of the GECO mechanism to generate a new text from the embedding of the selected query.
            :type corrector_model_name: str
            :param cache_path_pins: the file path where the embeddings of the query pool will be cached. This is used to speed up the initialization process by avoiding re-encoding the queries in the pool every time the GECO class is instantiated. If the cache file exists and contains embeddings for the same number of queries as the loaded pool, it will be loaded directly. Otherwise, the queries will be re-encoded andthe cache will be updated.
            :type cache_path_pins: str
            :param collection_name: the name ofthe collection to load queries from. Defaults to "msmarco-passage/train". This collection should be compatible withthe ir_datasets library and should contain a set of queries that will be used asthe pool forthe exponential mechanism. The queries will be embedded usingthe sentence transformer model and used for selection inthe obfuscation process.
            :type collection_name: str, optional
            :param epsilons: a list of epsilon values to use for obfuscation. This is not used directly in the initialization but can be stored for reference or used in specific implementations of the GECO mechanism that may want to precompute certain values based on the epsilon values. Defaults to a list of common epsilon values used in differential privacy.
            :type epsilons: List[float], optional
            :param kwargs: additional keyword arguments, such as *top_percentile* and *encode_batch_size* which can be used to configurethe behavior ofthe GECO mechanism. top_percentile determinesthe percentile threshold for filtering candidates inthe exponential mechanism based on their utility, and encode_batch_size determinesthe batch size to use when encodingthe queries inthe pool, which can affectthe speed and memory usage ofthe encoding process.
            :type kwargs: additional keyword arguments  

        """
        #initialisation
        if log is None:
            log = geco.utils.createLogger("GECO_Obfuscator_Logger")
        self.log = log

        assert st_model_name is not None, "st_model_name cannot be None"
        self.st_model_name = st_model_name
        assert corrector_model_name is not None, "corrector_model_name cannot be None"
        self.corrector_model_name = corrector_model_name
        assert inversion_model_name is not None, "inversion_model_name cannot be None"
        self.inversion_model_name = inversion_model_name
        self._load_models()
        
        self.collection_name = collection_name
        self.cache_path_pins = cache_path_pins
        self.top_percentile = kwargs.get("top_percentile", 0.001)
        assert 0.0 < self.top_percentile <= 1.0, "top_percentile must be in the range (0.0, 1.0]"
        self.encode_batch_size = kwargs.get("encode_batch_size", 512)
        assert self.encode_batch_size > 0, "encode_batch_size must be a positive integer"
        queries = self._load_pool()
        self.queries, self.embeddings = self._load_or_build_pool_embeddings(queries)

        self.n_pool = len(self.embeddings)
        self.utility_function = None #to be defined in the specific implementations of GECO (e.g. CosineGECO, KernelDensityGECO, CorrelationGECO)
        self.global_sensitivity = None #to be defined in the specific implementations of GECO (e.g. CosineGECO, KernelDensityGECO, CorrelationGECO)
        self.epsilons = epsilons
        assert all(epsilon > 0 for epsilon in self.epsilons), "All epsilon values must be strictly positive"
        self.log.info(f"Epsilon values for obfuscation: {self.epsilons}")

        self.additional_kwargs = kwargs

    def __str__(self):
        return f"GECO Obfuscator (Abstract) with st_model={self.st_model_name}, inversion_model={self.inversion_model_name}, corrector_model={self.corrector_model_name}, top_percentile={self.top_percentile}, n_corpus={self.n_pool}, epsilons={self.epsilons}.\n WARNING: No utility function specified, this is an abstract class. Please use a specific implementation of GECO with a defined utility function (e.g. CosineGECO, KernelDensityGECO, CorrelationGECO) for actual obfuscation."
    
    def _load_models(self) -> None:
        """
        Method to load all required models (embedder and generator) for the obfuscation mechanism. This method is called during the initialization of the GECO class to ensure that all models are loaded and ready for use when obfuscating texts.
        """
        #Embedder Model
        self.log.info(f"Loading sentence transformer model for embedding: {self.st_model_name}")
        st_model = SentenceTransformer(self.st_model_name)

        self.log.info(f"Loading vec2text inversion (zero-step) model: {self.inversion_model_name}")
        inversion_model = vec2text.models.InversionModel.from_pretrained(self.inversion_model_name)

        self.log.info(f"Loading vec2text corrector model: {self.corrector_model_name}")
        corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained(self.corrector_model_name)

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
        self.log.info("Vec2Text correctors loaded successfully.")
        self.st_model, self.corrector = st_model, corrector

    def _load_pool(self) -> List[str]:
        """
        Load a pool (of queries) from a specified collection. This pool will be used in the exponential mechanism to select a query based on the utility function. The queries are loaded from a collection compatible with the ir_datasets library, and they are expected to be in a format where each query has a 'text' field that contains the query text. The loaded queries are returned as a list of strings.

        :return: a list of queries loaded from the specified collection.
        :rtype: List[str]
        """
        try:
            import ir_datasets
        except ImportError as exc:
            raise ImportError("Missing IR datasets library. Please install it with: pip install ir-datasets") from exc
        self.log.info("Loading pool of queries")
        ds = ir_datasets.load(self.collection_name)
        dataset = ds.queries_iter()
        df: pd.DataFrame = pd.DataFrame(dataset)  
        queries: List[str] = df['text'].tolist()

        self.log.info(f"Loaded {len(queries):,} queries.")
        return queries

    def _load_or_build_pool_embeddings(self, queries: List[str]) -> Tuple[List[str], torch.Tensor]:
        """
        Load or build embeddings for the query pool. This method checks if a cache file exists at the specified path and if it contains embeddings for the same number of queries as the loaded pool. If the cache is valid, it loads the queries and their corresponding embeddings from the cache. If the cache is not valid (e.g., it does not exist or has a size mismatch), it encodes the queries using the sentence transformer model to generate their embeddings, saves them to the cache for future use, and returns the queries along with their embeddings.
        """
        cache: Path = Path(self.cache_path_pins)
        if cache.exists():
            self.log.info(f"Loading embedding cache")
            data: object = torch.load(self.cache_path_pins, map_location="cpu")
            cq: List[str] = data["queries"]
            ce: torch.Tensor = data["embeddings"]
            if len(cq) == len(queries):
                self.log.info(f"Cache valid: {len(cq):,} queries x {ce.shape[1]} dimensions")
                return cq, ce
            self.log.error(f"Cache size mismatch ({len(cq):,} vs {len(queries):,}). Re-encoding needed.")

        self.log.info(f"Encoding {len(queries):,} queries (batch={self.encode_batch_size}).")
        embeddings:object = self.st_model.encode(sentences = queries, 
                                          batch_size=self.encode_batch_size, 
                                          convert_to_tensor=True, 
                                          normalize_embeddings=True, 
                                          show_progress_bar=True).float().cpu()

        torch.save({"queries": queries, "embeddings": embeddings}, self.cache_path_pins)
        self.log.info(f"Cache saved to {self.cache_path_pins}.")
        return queries, embeddings
    
    def check_utility_function(self) -> None:
        """
        Check Utility Function for debugging and repro. This function logs the Utiltiy function.
        """
        if self.utility_function is None:
            self.log.warning("Utility function is not defined. This is an abstract GECO obfuscator. Please use a specific implementation of GECO with a defined utility function (e.g. CosineGECO, KernelDensityGECO, CorrelationGECO) for actual obfuscation.")
        else:
            self.log.info(f"Using utility function: {self.utility_function}.")

    def check_global_sensitivity(self) -> None:
        """
        Check Global Sensitivity for debugging and repro. This function logs the Global Sensitivity.
        """
        if self.global_sensitivity is None:
            self.log.warning("Global sensitivity is not defined. This is an abstract GECO obfuscator. Please use a specific implementation of GECO with a defined global sensitivity (e.g. CosineGECO, KernelDensityGECO, CorrelationGECO) for actual obfuscation.")
        else:
            self.log.info(f"Global Sensitivity: {self.global_sensitivity}.")

    def set_utility_function(self, utility_function: str) -> None:
        """
        Set the utility function for the GECO obfuscator. This function allows to set the utility function for the obfuscator, which will be used in the exponential mechanism to select a query from the pool based on its utility with respect to the input embedding. The utility function should be defined in the specific implementations of GECO (e.g. CosineGECO, KernelDensityGECO, CorrelationGECO) and should be compatible with the way the utility is computed in those implementations.

        :param utility_function: the name of the utility function to set for the obfuscator.
        :type utility_function: str
        """
        self.utility_function = utility_function
        self.log.info(f"Utility function set to: {self.utility_function}.")

    def set_global_sensitivity(self, global_sensitivity: float) -> None:
        """
        Set the global sensitivity for the GECO obfuscator. This function allows to set the global sensitivity for the obfuscator, which is a parameter used in the exponential mechanism to determine how much noise to add when selecting a query from the pool based on its utility. The global sensitivity should be defined in the specific implementations of GECO (e.g. CosineGECO, KernelDensityGECO, CorrelationGECO) and should be compatible with the way the utility is computed in those implementations.

        :param global_sensitivity: the value of the global sensitivity to set for the obfuscator.
        :type global_sensitivity: float
        """
        self.global_sensitivity = global_sensitivity
        self.log.info(f"Global sensitivity set to: {self.global_sensitivity}.")

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