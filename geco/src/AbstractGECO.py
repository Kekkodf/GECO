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

    def __init__(self,
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
        return f"GECO Obfuscator (Abstract) with st_model={self.st_model_name}, inversion_model={self.inversion_model_name}, corrector_model={self.corrector_model_name}, top_percentile={self.top_percentile}, n_pool={self.n_pool}, epsilons={self.epsilons}.\n WARNING: No utility function specified, this is an abstract class. Please use a specific implementation of GECO with a defined utility function (e.g. CosineGECO, KernelDensityGECO, CorrelationGECO) for actual obfuscation."
    
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

    def _load_or_build_pool_embeddings(self, 
                                       queries: List[str]) -> Tuple[List[str], torch.Tensor]:
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
    
    def check_utility_function(self) -> bool:
        """
        Check Utility Function for debugging and repro. This function logs the Utiltiy function.
        """
        if self.utility_function is None:
            self.log.warning("Utility function is not defined. This is an abstract GECO obfuscator. Please use a specific implementation of GECO with a defined utility function (e.g. CosineGECO, KernelDensityGECO, CorrelationGECO) for actual obfuscation.")
            return False
        else:
            self.log.info(f"Using utility function: {self.utility_function}.")
            return True

    def check_global_sensitivity(self) -> bool:
        """
        Check Global Sensitivity for debugging and repro. This function logs the Global Sensitivity.
        """
        if self.global_sensitivity is None:
            self.log.warning("Global sensitivity is not defined. This is an abstract GECO obfuscator. Please use a specific implementation of GECO with a defined global sensitivity (e.g. CosineGECO, KernelDensityGECO, CorrelationGECO) for actual obfuscation.")
            return False
        else:
            self.log.info(f"Global Sensitivity: {self.global_sensitivity}.")
            return True

    def set_utility_function(self, 
                             utility_function: str) -> None:
        """
        Set the utility function for the GECO obfuscator. This function allows to set the utility function for the obfuscator, which will be used in the exponential mechanism to select a query from the pool based on its utility with respect to the input embedding. The utility function should be defined in the specific implementations of GECO (e.g. CosineGECO, KernelDensityGECO, CorrelationGECO) and should be compatible with the way the utility is computed in those implementations.

        :param utility_function: the name of the utility function to set for the obfuscator.
        :type utility_function: str
        """
        self.utility_function = utility_function
        self.log.info(f"Utility function set to: {self.utility_function}.")

    def set_global_sensitivity(self, 
                               global_sensitivity: float) -> None:
        """
        Set the global sensitivity for the GECO obfuscator. This function allows to set the global sensitivity for the obfuscator, which is a parameter used in the exponential mechanism to determine how much noise to add when selecting a query from the pool based on its utility. The global sensitivity should be defined in the specific implementations of GECO (e.g. CosineGECO, KernelDensityGECO, CorrelationGECO) and should be compatible with the way the utility is computed in those implementations.

        :param global_sensitivity: the value of the global sensitivity to set for the obfuscator.
        :type global_sensitivity: float
        """
        self.global_sensitivity = global_sensitivity
        self.log.info(f"Global sensitivity set to: {self.global_sensitivity}.")

    def embed(self, 
              texts: List[str]) -> torch.Tensor:
        """
        Embed a list of texts using the sentence transformer model.

        :param texts: a list of input texts to embed.
        :type texts: List[str]

        :return: a tensor containing the embeddings of the input texts, shape (len(texts), embedding_dim).
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            embeddings = self.st_model.encode(
                sentences=texts,
                batch_size=self.encode_batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return embeddings.float()

    def obfuscate(self,
                  text: str,
                  epsilon: float,
                  seed: Optional[int] = None,
                  top_k_report: int = 5) -> Tuple[str, str, float, float, float, List[Tuple[str, float, float]]]:
        """
        Obfuscate a single input text using the GECO mechanism. This method takes an input text, encodes it into an embedding, uses the exponential mechanism to select a query from the pool based on the utility function, reconstructs a new text from the embedding of the selected query using the corrector model, and returns the obfuscated text along with diagnostics such as cosine similarity between the input and output embeddings, and the top candidates from the exponential mechanism.

        :param text: the input text to obfuscate.
        :type text: str
        :param epsilon: the privacy budget to use for the exponential mechanism when selecting a query from the pool. This value determines the level of privacy provided by the obfuscation, with smaller values providing stronger privacy guarantees but potentially lower utility of the obfuscated text.
        :type epsilon: float
        :param seed: an optional random seed for reproducibility. This seed will be used in the exponential mechanism to ensure that the selection of the query from the pool is reproducible across runs with the same input and parameters. If None, the selection will be non-deterministic.
        :type seed: Optional[int]
        :param top_k_report: the number of top candidates to report from the exponential mechanism for diagnostics. This determines how many of the top candidates (queries) from the exponential mechanism will be included in the output for diagnostic purposes, along with their probabilities and utilities. Defaults to 5.
        :type top_k_report: int, optional

        :return: a tuple containing the obfuscated text, the selected query, the cosine similarity between the input and output embeddings, the cosine similarity between the input embedding and the selected query embedding, the angle (in degrees) between the input and output embeddings, and a list of the top candidates from the exponential mechanism with their probabilities and utilities.
        :rtype: Tuple[str, str, float, float, float, List[Tuple[str, float, float]]]

        The returned tuple contains the following elements:
        - obfuscated_text: the generated obfuscated text from the GECO mechanism.
        - selected_query: the query from the pool that was selected by the exponential mechanism as the pin for reconstruction.
        - cos_io: the cosine similarity between the input embedding and the output embedding, which provides a measure of how similar the obfuscated text is to the original text in the embedding space.
        - cos_iq: the cosine similarity between the input embedding and the selected query embedding, which provides a measure of how similar the selected query is to the original text in the embedding space.
        - angle: the angle in degrees between the input embedding and the output embedding, which provides another measure of similarity between the original and obfuscated texts in the embedding space (smaller angles indicate more similar embeddings).
        - top_candidates: a list of the top candidates from the exponential mechanism, where each candidate is a tuple containing the query string, its probability of selection, and its utility value. This list provides insight into which queries were considered most relevant by the exponential mechanism and how they compare in terms of utility and selection probability.          

        Example usage:
        >>> obfuscator = GECOObfuscator(log=logger, st_model_name="all-MiniLM-L6-v2", inversion_model_name="inversion-model", corrector_model_name="corrector-model", cache_path_pins="cache/pins.pt")
        >>> obfuscated_text, selected_query, cos_io, cos_iq, angle, top_candidates = obfuscator.obfuscate("This is a sensitive text that needs to be obfuscated.", epsilon=10.0, seed=42, top_k_report=5)
        """
        self.log.info(f"Obfuscating text: '{text}' with epsilon={epsilon}.")
        # encoding input text
        e = self.embed([text])[0]

        # exponential mechanism to select the pin
        check = self.check_utility_function() and self.check_global_sensitivity()
        if not check:
            self.log.error("Utility function and global sensitivity must be defined for the obfuscation to work. Please use a specific implementation of GECO with a defined utility function and global sensitivity (e.g. CosineGECO, KernelDensityGECO, CorrelationGECO) for actual obfuscation.")
            raise ValueError("Utility function and global sensitivity must be defined for the obfuscation to work. Please use a specific implementation of GECO with a defined utility function and global sensitivity (e.g. CosineGECO, KernelDensityGECO, CorrelationGECO) for actual obfuscation.")
        selected_query, selected_emb, probs, utilities = self.exponential_mechanism(e, epsilon, seed=seed)

        # Generation of the output text
        self.log.info(f"Pin query selected by the exponential mechanism: '{selected_query}'")
        output_text = self.generate_from_embedding(selected_emb)
        self.log.info(f"Generated obfuscated text: '{output_text}'")

        # Diagnostics
        out_emb = self.embed([output_text])[0]
        cos_io  = float((e.cpu() @ out_emb.cpu()).item())
        cos_iq  = float((e.cpu() @ selected_emb.cpu()).item())
        angle   = float(np.degrees(np.arccos(np.clip(cos_io, -1.0, 1.0))))

        # Rebuild filtered_idx for the top-k report
        e_cpu = (e / e.norm()).cpu()
        chunk = 50_000; N = self.n_pool
        all_utils = np.empty(N, dtype=np.float32)
        for s in range(0, N, chunk):
            ee = min(s + chunk, N)
            all_utils[s:ee] = (self.embeddings[s:ee] @ e_cpu).numpy()
        if self.top_percentile < 1.0:
            cut = float(np.percentile(all_utils, 100.0 * (1.0 - self.top_percentile)))
            fidx = np.where(all_utils >= cut)[0]
        else:
            fidx = np.arange(N)

        top_local = np.argsort(probs)[::-1][:top_k_report]
        top_candidates = [
            (self.queries[fidx[i]], float(probs[i]), float(utilities[i]))
            for i in top_local
        ]

        return (output_text, selected_query, cos_io, cos_iq, angle, top_candidates
        )

    def batch_obfuscate(self, 
                        texts: List[str], 
                        epsilon: float, 
                        seed: Optional[int] = None) -> List[Tuple[str, str, float, float, float, List[Tuple[str, float, float]]]]:
        return [self.obfuscate(t, epsilon, seed=seed) for t in texts]
    
    def generate_from_embedding(self,
                                target_emb: torch.Tensor,
                                num_corrector_steps: int = 1) -> str:
        """
        Last part of the GECO mechanism: generation of the obfuscated text from the embedding of the selected query using the corrector model. This method takes the target embedding (the embedding of the selected query) and uses the vec2text corrector model to generate a new text that corresponds to that embedding. The generation process can be configured with the number of steps to use in the corrector model, which can affect the quality and diversity of the generated text. The method returns the generated obfuscated text as a string.   

        :param target_emb: the embedding of the selected query from the exponential mechanism, which will be used as the target for generation. This embedding should be a tensor of shape (embedding_dim,) and should be compatible with the corrector model's expected input.
        :type target_emb: torch.Tensor
        :param num_corrector_steps: the number of steps to use in the corrector model for generating the obfuscated text. This parameter can affect the quality and diversity of the generated text, with more steps potentially leading to better generation at the cost of increased computation time. Defaults to 1.
        :type num_corrector_steps: int, optional

        :return: the generated obfuscated text corresponding to the target embedding.
        :rtype: str
        """
        device = next(self.corrector.model.parameters()).device
        with torch.no_grad():
            texts = vec2text.invert_embeddings(
                embeddings=target_emb.unsqueeze(0).to(device),
                corrector=self.corrector,
                num_steps=num_corrector_steps,
            )
        return texts[0].strip()
    
    def exponential_mechanism(self,
                              e: torch.Tensor,
                              epsilon: float,
                              seed: Optional[int] = None) -> Tuple[str, torch.Tensor, np.ndarray, np.ndarray]:
        """
        Exponential mechanism to select a query from the pool based on the utility function. This method computes the utility of each query in the pool with respect to the input embedding, applies a percentile filter to keep only the top candidates, and then uses the Gumbel-max trick to sample a query from the filtered candidates according to the exponential mechanism distribution. The method returns the selected query, its embedding, and diagnostic information about the probabilities and utilities of the candidates.

        :param e: the input embedding of the text to obfuscate, shape (d,).
        :type e: torch.Tensor
        :param epsilon: the privacy budget to use for the exponential mechanism. This value determines the level of privacy provided by the selection of the query, with smaller values providing stronger privacy guarantees but potentially lower utility of the selected query.
        :type epsilon: float
        :param seed: an optional random seed for reproducibility. This seed will be used in the Gumbel-max trick to ensure that the selection of the query from the filtered candidates is reproducible across runs with the same input and parameters. If None, the selection will be non-deterministic.
        :type seed: Optional[int]
        
        :return: a tuple containing the selected query, its embedding, the probabilities of the candidates, and their utilities. The selected query is the one chosen by the exponential mechanism based on the input embedding and the specified epsilon, and its embedding is the corresponding embedding from the pool. The probabilities and utilities provide diagnostic information about the candidates that were considered in the selection process.
        :rtype: Tuple[str, torch.Tensor, np.ndarray, np.ndarray]
        """
        if not (0.0 < self.top_percentile <= 1.0):
            raise ValueError(f"top_percentile must be in (0, 1], got {self.top_percentile}")
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed()

        e_cpu = (e / e.norm()).cpu()
        N = len(self.queries)

        # Compute utilities in chunks to bound peak RAM
        chunk = 50_000
        all_utils = np.empty(N, dtype=np.float32)
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            all_utils[start:end] = (self.embeddings[start:end] @ e_cpu).numpy()

        # Percentile filter
        cutoff = float(np.percentile(all_utils, 100.0 * (1.0 - self.top_percentile)))
        mask = all_utils >= cutoff

        # Get indices of candidates that passed the filter
        filtered_idx = np.where(mask)[0]
        utilities = all_utils[filtered_idx]
        K = len(filtered_idx)
        self.log.info(f"Using top_percentile {self.top_percentile:.2f}, meaning {K:,}/{N:,} candidates.")
        self.log.info(f"(u in [{utilities.min():.4f}, {utilities.max():.4f}])")

        # Gumbel-max trick (equivalent to sampling from Exp Mech distribution)
        # ref. https://differentialprivacy.org/one-shot-top-k/
        gumbel_scale = (2.0 * self.global_sensitivity) / epsilon
        noisy = utilities + np.random.gumbel(0.0, gumbel_scale, size=K)
        local_best = int(np.argmax(noisy))
        global_best = int(filtered_idx[local_best])

        # True Exp Mech probabilities for diagnostics (not used in sampling)
        log_w = (epsilon / (2.0 * self.global_sensitivity)) * utilities
        log_w -= log_w.max()
        probs = np.exp(log_w); probs /= probs.sum()

        return self.queries[global_best], self.embeddings[global_best], probs, utilities
