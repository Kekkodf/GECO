import numpy as npq
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from sentence_transformers import SentenceTransformer
import vec2text
import warnings
import numpy as np
from typing import List
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')

class GECOvec2text:
    '''GECOvec2text is a class that implements the GECO algorithm using vector-to-text models for encoding and correction.
    It utilizes a pre-trained encoder to convert input data into vector representations and a corrector to invert the embeddings based on the vec2text strategy.
    Attributes:
        encoder_model (PreTrainedModel): The pre-trained model used for encoding input data into vector representations.
        encoder_tokenizer (PreTrainedTokenizer): The tokenizer associated with the encoder model for processing input text.
        corrector (object): The corrector object responsible for inverting the embeddings based on the vec2text strategy.
        epsilon (float): The epsilon value used in the GECO algorithm to control the trade-off between reconstruction error and regularization.
    '''
    def __init__(self, logger:object, epsilon:float=1.0) -> None:

        '''
        
        '''
        self.logger = logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to(device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
        self.corrector = vec2text.load_pretrained_corrector("gtr-base")
        self.epsilon = 1.0

    def get_gtr_embeddings(self, text_list:list) -> torch.Tensor:
        '''get_gtr_embeddings is a method that takes a list of text inputs and returns their corresponding vector embeddings using the GTR encoder.
        Args:
            text_list (list): A list of strings for which to compute the embeddings.
        Returns:
            ...
        '''
        inputs = self.tokenizer(text_list,
                       return_tensors="pt",
                       max_length=128,
                       truncation=True,
                       padding="max_length",).to(self.device)

        with torch.no_grad():
            model_output = self.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            hidden_state = model_output.last_hidden_state
            embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])
        noise = get_noise(embeddings.shape, self.epsilon)
        return embeddings + noise, noise

    def reconstruct(self, embeddings, num_steps) -> List[str]:

        return vec2text.invert_embeddings(embeddings, self.corrector, num_steps=num_steps)
