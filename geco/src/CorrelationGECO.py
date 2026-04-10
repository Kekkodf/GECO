import numpy as np
import torch
from tabulate import tabulate

from geco.src import AbstractGECO


class CorrelationGECO(AbstractGECO.Obfuscator):
    """
    Correlation-based GECO obfuscator.
    -------------------------
    Extends the AbstractGECO Obfuscator class and implements the correlation as the utility function for measuring the similarity between the query embedding and the pool embeddings.
    
    The global sensitivity is set to 2.0 for the correlation utility function.

    The main steps of the GECO mechanism are as follows:
    1. Encode the input text into an embedding using a sentence transformer model.
    2. Use the exponential mechanism to select a query from a pool of queries based on a utility function (e.g. cosine similarity)
    3. Reconstruct a new text from the embedding of the selected query using a corrector model.

    The CorrelationGECO class inherits from the AbstractGECO Obfuscator class and provides methods for obfuscating a single text or a batch of texts, and it also includes diagnostics such as cosine similarity between the input and output embeddings, and the top candidates from the exponential mechanism.
    """
    def __init__(self, **kwargs):
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
            :param kwargs: additional keyword arguments, such as top_percentile and encode_batch_size, which can be used to configurethe behavior ofthe GECO mechanism. top_percentile determinesthe percentile threshold for filtering candidates inthe exponential mechanism based on their utility, and encode_batch_size determinesthe batch size to use when encodingthe queries inthe pool, which can affectthe speed and memory usage ofthe encoding process.
            :type kwargs: additional keyword arguments  

        """
        
        super().__init__(**kwargs)
        self.utility_function = 'correlation'
        self._utility_fn = self._compute_utility
        self.global_sensitivity = 2.0
        self.log.info(f"CorrelationGECO Obfuscator initialized with utility function: {self.utility_function} and global sensitivity: {self.global_sensitivity}.")

    def __str__(self):
        COL1, COL2 = 22, 60
        def row(label, value):
            return [f"{label:<{COL1}}", f"{str(value):<{COL2}}"]

        model_section: str = tabulate(
            [
                row("📡  Logger",          self.log.handlers[0].baseFilename),
                row("🧠  Sentence Model",  self.st_model_name),
                row("🔁  Inversion Model", self.inversion_model_name),
                row("🔧  Corrector Model", self.corrector_model_name),
                row("📊  Top Percentile",  self.top_percentile),
                row("📂  Collection Pool", self.collection_name),
                row("🎱  Pool Size",       self.n_pool),
                row("🔑  Epsilons",        self.epsilons),
                row("📐  Utility Function",    self.utility_function),
                row("📏  Global Sensitivity",  self.global_sensitivity)
            ],
            headers=[f"{'Parameter':<{COL1}}", f"{'Value':<{COL2}}"],
            tablefmt="rounded_outline",
        )


        return "\n".join([
            "🔒  GECO Obfuscator  (CorrelationGECO)\n",
            model_section
        ])

    def _compute_utility(self, query_embedding, pool_embeddings) -> torch.Tensor:
        """Pearson correlation between query_embedding and each of the pool_embeddings. The correlation is computed as the cosine similarity between the mean-centered query_embedding and the mean-centered pool_embeddings. The mean-centering is done by subtracting the mean of the embeddings from each embedding, which ensures that the correlation is not affected by the magnitude of the embeddings but only by their direction and relative position in the embedding space (Remark: Should be already normalised). The global sensitivity for the correlation utility function is 2.0, which means that changing one query in the pool can change the utility by at most 2.0.

        :param query_embedding: the embedding of the input text for which we want to compute the utility of the pool queries. This should be a 1D numpy array or PyTorch tensor of the same dimension as the pool embeddings.
        :type query_embedding: np.ndarray or torch.Tensor
        :param pool_embeddings: the embeddings of the queries in the pool. This should be a 2D numpy array or PyTorch tensor of shape (n_pool, embedding_dim), where n_pool is the number of queries in the pool and embedding_dim is the dimension of the embeddings.
        :type pool_embeddings: np.ndarray or torch.Tensor

        :return: the utility of each query in the pool with respect to the input query embedding, computed as the correlation. This should be a 1D numpy array or PyTorch tensor of shape (n_pool,).
        :rtype: np.ndarray or torch.Tensor
        """

        b_centered = query_embedding - query_embedding.mean()
        b_norm = np.linalg.norm(b_centered)
        if b_norm == 0:
            self.log.warning("Query embedding has zero variance, returning zero utility for all pool queries.")
            raise RuntimeError("Query embedding has zero variance, cannot compute correlation utility.")
        a_centered = pool_embeddings - pool_embeddings.mean(axis=1, keepdims=True)
        a_norms = np.linalg.norm(a_centered, axis=1)
        # Avoid division by zero in case of zero variance in pool embeddings
        zero_variance_mask = a_norms == 0
        if np.any(zero_variance_mask):
            self.log.warning(f"{np.sum(zero_variance_mask)} pool embeddings have zero variance, setting their utility to zero.")
            raise RuntimeError(f"{np.sum(zero_variance_mask)} pool embeddings have zero variance, cannot compute correlation utility.")
        utilities = (a_centered @ b_centered) / (a_norms * b_norm)
        return utilities