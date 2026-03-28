import pandas as pd
import torch
import transformers
import vec2text
import vec2text.models
import vec2text.trainers
import vec2text.collator
import logging

from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer


def load_gtr_corrector(
        log: logging.Logger,
        st_model_name: str = "sentence-transformers/gtr-t5-base",
        inversion_model: str = "ielabgroup/vec2text_gtr-base-st_inversion",
        corrector_model: str = "ielabgroup/vec2text_gtr-base-st_corrector", 
        **kwargs) -> Tuple[SentenceTransformer, vec2text.trainers.Corrector]:
    """
    This method is an helper function used to instantiate the (gtr-t5-base) model, inversion and corrector used for the generation of the privatised text.
    We suggest to use the same model for encoding and reconstruction: if you need to train your own corrector, follow the instructions in the vec2text repository and then simply replace the model name in the corrector_model argument.
    The st_model_name can be replaced with any sentence transformer model, but we suggest to use gtr-t5-base for better performance of the obfuscation mechanism, as it is the same model trained as the corrector.

    :param log: logger to use for logging the loading process.
    :type log: logging.Logger
    :param st_model_name: name of the sentence transformer model to use for embedding. Defaults to "sentence-transformers/gtr-t5-base".
    :type st_model_name: str, optional
    :param inversion_model: name of the vec2text inversion model to use. Defaults to "ielabgroup/vec2text_gtr-base-st_inversion".
    :type inversion_model: str, optional
    :param corrector_model: name of the vec2text corrector model to use. Defaults to "ielabgroup/vec2text_gtr-base-st_corrector".
    :type corrector_model: str, optional
    :param kwargs: additional keyword arguments.
    :type kwargs: additional keyword arguments

    :return: a tuple containing the loaded sentence transformer model and the vec2text corrector.
    :rtype: Tuple[SentenceTransformer, vec2text.trainers.Corrector]

    Example usage:
    >>> log = create('mylogger')
    >>> st_model, corrector = load_gtr_corrector(log)
    """
    # Loading the sentence transformer model used for encoding the input text
    try:
        log.info(f"Loading sentence transformer model: {st_model_name}")
        st_model = SentenceTransformer(st_model_name)
    except Exception as e:
        log.error(f"Error loading sentence transformer model: {e}")
        raise e
    # Loading the vec2text inversion and corrector models, as implemented by Morris et al.
    # security checks
    assert inversion_model in ["jxm/gtr__nq__32", "ielabgroup/vec2text_gtr-base-st_inversion"]
    log.info(f"Loading vec2text inversion model: {inversion_model}")
    inversion_model = vec2text.models.InversionModel.from_pretrained(
        inversion_model
    )
    assert corrector_model in ["jxm/gtr__nq__32__correct", "ielabgroup/vec2text_gtr-base-st_corrector"]
    log.info(f"Loading vec2text corrector model: {corrector_model}")
    corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained(
        corrector_model
    )

    # we instantiate the vec2text inversion trainer with dummy datasets and a data collator, as implememnted by Morris et al.
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

    # we instantiate the vec2text pre-trained corrector using the inversion trainer's tokenizer, as implememnted by Morris et al.
    corrector = vec2text.trainers.Corrector(
        model=corrector_model,
        inversion_trainer=inversion_trainer,
        args=None,
        data_collator=vec2text.collator.DataCollatorForCorrection(
            tokenizer=inversion_trainer.model.tokenizer
        ),
    )
    
    log.info("Corrector loaded.")
    return st_model, corrector

def embed(
        log: logging.Logger,
        st_model: SentenceTransformer,
        texts: List[str], 
        **kwargs) -> torch.Tensor:
    """
    Return (N, st_model.get_sentence_embedding_dimension()) float32 L2-normalised embeddings from st_model.

    :param log: logger to use for logging the embedding process.
    :type log: logging.Logger
    :param st_model: the sentence transformer model to use for embedding the input texts.
    :type st_model: SentenceTransformer
    :param texts: the list of input texts to embed, N is the number of input texts.
    :type texts: List[str]
    :param kwargs: additional keyword arguments.
    :type kwargs: additional keyword arguments

    :return: a tensor containing the L2-normalised embeddings of the input texts.
    :rtype: torch.Tensor

    Example usage:
    >>> log = create('mylogger')
    >>> st_model, _ = load_gtr_corrector(log)
    >>> texts = ["This is a test sentence.", "Another test sentence."]
    >>> embeddings = embed(log, st_model, texts)
    """
    with torch.no_grad():
        embs = st_model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    
    log.info(f"Encoded {len(texts)} texts into embeddings of shape {embs.shape}.")
    
    return embs.float()

def load_pool_of_queries(
        log: logging.Logger, 
        collection_name: str = "msmarco-passage/train", 
        **kwargs) -> List[str]:
    """
    Load a pool of queries from a specified collection.

    :param log: logger to use for logging the loading process.
    :type log: logging.Logger
    :param collection_name: the name of the collection to load queries from. Defaults to "msmarco-passage/train".
    :type collection_name: str, optional
    :param kwargs: additional keyword arguments.
    :type kwargs: additional keyword arguments

    :return: a list of queries loaded from the specified collection.
    :rtype: List[str]

    Example usage:
    >>> log = create('mylogger')
    >>> queries = load_pool_of_queries(log)
    """
    try:
        import ir_datasets
    except ImportError as exc:
        raise ImportError("Missing IR datasets library. Please install it with: pip install ir-datasets") from exc
    log.info("Loading MS MARCO training queries ...")
    ds = ir_datasets.load(collection_name)
    dataset = ds.queries_iter()
    df: pd.DataFrame = pd.DataFrame(dataset)  
    queries: List[str] = df['text'].tolist()
    
    log.info(f"Loaded {len(queries):,} queries.")
    return queries


def build_or_load_pool_of_queries_embeddings(
        log: logging.Logger,
        st_model: SentenceTransformer,
        queries: List[str],
        cache_path: str = "./query_pool/msmarco_gtr_embeddings.pt",
        encode_batch_size: int = 512, 
        **kwargs) -> Tuple[List[str], torch.Tensor]:
    """
    Build or load embeddings for a pool of queries.

    :param log: logger to use for logging the embedding process.
    :type log: logging.Logger
    :param st_model: the sentence transformer model to use for embedding the queries.
    :type st_model: SentenceTransformer
    :param queries: the list of queries to embed.
    :type queries: List[str]
    :param cache_path: the path to the cache file for storing query embeddings. Defaults to "./query_pool/msmarco_gtr_embeddings.pt".
    :type cache_path: str, optional
    :param encode_batch_size: the batch size to use for encoding the queries. Defaults to 512.
    :type encode_batch_size: int, optional
    :param kwargs: additional keyword arguments.
    :type kwargs: additional keyword arguments

    :return: a tuple containing the list of queries and their corresponding embeddings.
    :rtype: Tuple[List[str], torch.Tensor]

    Example usage:
    >>> log = create('mylogger')
    >>> st_model, _ = load_gtr_corrector(log)
    >>> queries = load_pool_of_queries(log)
    >>> queries, embeddings = build_or_load_pool_of_queries_embeddings(log, st_model, queries)
    """
    cache: Path = Path(cache_path)
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