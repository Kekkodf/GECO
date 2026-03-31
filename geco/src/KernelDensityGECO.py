from geco.src import AbstractGECO

class KernelDensityGECO(AbstractGECO.Obfuscator):
    """Radial Basis Function kernel (RBF kernel) based GECO obfuscator.
    -------------------------
    Extends the AbstractGECO Obfuscator class and implements the RBF kernel as the utility function for measuring the similarity between the query embedding and the pool embeddings.
    
    The global sensitivity is set to 1.0 for the RBF kernel. The gamma hyperparameter for the RBF kernel is set to 1.0 by default (Conservative).

    The main steps of the GECO mechanism are as follows:
    1. Encode the input text into an embedding using a sentence transformer model.
    2. Use the exponential mechanism to select a query from a pool of queries based on a utility function (e.g. cosine similarity)
    3. Reconstruct a new text from the embedding of the selected query using a corrector model.

    The KerenlDensity class inherits from the AbstractGECO Obfuscator class and provides methods for obfuscating a single text or a batch of texts, and it also includes diagnostics such as cosine similarity between the input and output embeddings, and the top candidates from the exponential mechanism.
    """
    def __init__(self, **kwargs):
        """        
        Initialisation of the GECO obfuscator. 
        
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
        self.utility_function = 'rbf'
        self.global_sensitivity = 1.0
        self.gamma = 1.0  # Hyperparameter for RBF kernel
        self.log.info(f"Gamma for RBF kernel in KernelDensityGECO: {self.gamma}")
        self.log.info(f"KernelDensityGECO Obfuscator initialized with utility function: {self.utility_function} and global sensitivity: {self.global_sensitivity}")

    def __str__(self):
        return f"KernelDensityGECO Obfuscator with st_model={self.st_model_name}, inversion_model={self.inversion_model_name}, corrector_model={self.corrector_model_name}, top_percentile={self.top_percentile}, n_corpus={self.n_pool}, epsilon_values={self.epsilons}, utility_function={self.utility_function}, global_sensitivity={self.global_sensitivity}."
    

    
    def _compute_utility(self, query_embedding, pool_embeddings):
        ...
