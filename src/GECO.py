import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import vec2text
import warnings
from typing import List, Tuple
import math
import logging

warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
import torch._logging
torch._logging.set_logs(dynamo=logging.WARNING)


def get_noise(
    shape: tuple,
    epsilon: float,
    mechanism: str = 'laplace',
    sensitivity: float = 1.0,
    delta: float = 1e-5,
    device: str = 'cpu',
) -> torch.Tensor:
    '''
    Generate noise according to the specified DP mechanism.
    Supported mechanisms:
      - 'laplace': Isotropic noise with L1 sensitivity.
      - 'gaussian': Isotropic noise with L2 sensitivity, requires delta.
      - 'mahalanobis': Anisotropic noise aligned with embedding covariance.
    '''
    if epsilon <= 0:
        raise ValueError("epsilon must be strictly positive.")

    *batch_dims, d = shape

    if mechanism in ('laplace', 'mahalanobis'):
        scale = sensitivity / epsilon               
        raw = torch.randn(shape, device=device)    
        norms = raw.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        directions = raw / norms                  
        radii_np = np.random.gamma(
            shape=d,                              
            scale=scale,
            size=batch_dims or [1],
        )
        radii = torch.tensor(radii_np, dtype=torch.float32, device=device)
        if batch_dims:
            radii = radii.unsqueeze(-1)            # (..., 1) for broadcasting
        else:
            radii = radii.view(1, 1)

        noise = directions * radii

        if mechanism == 'mahalanobis':
            
            dim_weights = torch.linspace(1.0, 0.1, d, device=device)  
            dim_weights = dim_weights / dim_weights.norm()             
            noise = noise * dim_weights * math.sqrt(d)

        return noise

    elif mechanism == 'gaussian':
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1) for the Gaussian mechanism.")
        sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
        return torch.randn(shape, device=device) * sigma

    else:
        raise ValueError(f"Unknown mechanism '{mechanism}'. "
                         "Choose from 'laplace', 'gaussian', 'mahalanobis'.")


class GECOvec2text:
    '''
    GECO algorithm with differentially private embedding perturbation.

    The pipeline is:
      text -> GTR embedding -> +DP noise -> vec2text corrector -> obfuscated text

    The corrector is steered toward the noisy embedding, so reconstruction
    yields text that is semantically plausible but diverges from the original
    in proportion to the noise magnitude (i.e. inversely to epsilon).
    '''

    def __init__(
        self,
        logger: object,
        noise_mechanism: str = 'laplace',
        epsilon: float = 50.0,
        sensitivity: float = 1.0,
        delta: float = 1e-5,
    ) -> None:
        self.logger = logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = (
            AutoModel.from_pretrained("sentence-transformers/gtr-t5-base")
            .encoder.to(device=self.device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
        self.corrector = vec2text.load_pretrained_corrector("gtr-base")

        if epsilon <= 0:
            raise ValueError("Epsilon must be strictly positive.")
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.delta = delta
        self.noise_mechanism = noise_mechanism


    def set_epsilon(self, epsilon: float) -> None:
        if epsilon <= 0:
            self.logger.error("Epsilon must be strictly positive.")
            raise ValueError("Epsilon must be strictly positive.")
        self.epsilon = epsilon
        self.logger.info(f"Epsilon updated to {epsilon}.")

    def set_mechanism(self, mechanism: str) -> None:
        self.noise_mechanism = mechanism
        self.logger.info(f"Noise mechanism updated to {mechanism}.")


    def get_gtr_embeddings(self, text_list: list) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Encode texts, perturb embeddings with DP noise, return (noisy_emb, noise).
        The noise is returned separately so callers can audit the perturbation
        magnitude or reuse it for privacy-accounting purposes.
        '''
        inputs = self.tokenizer(
            text_list,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length",
        ).to(self.device)

        with torch.no_grad():
            model_output = self.encoder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
            )
            hidden_state = model_output.last_hidden_state
            embeddings = vec2text.models.model_utils.mean_pool(
                hidden_state, inputs['attention_mask']
            )

        noise = get_noise(
            shape=tuple(embeddings.shape),
            epsilon=self.epsilon,
            mechanism=self.noise_mechanism,
            sensitivity=self.sensitivity,
            delta=self.delta,
            device=self.device,
        )
        return embeddings + noise, noise

    def reconstruct(self, embeddings: torch.Tensor, num_steps: int = 20) -> List[str]:
        '''
        Invert noisy embeddings back to text via the vec2text corrector.
        num_steps controls the number of correction iterations: more steps
        push the output closer to a natural-language string that lives near
        the noisy embedding, not the original one.
        '''
        return vec2text.invert_embeddings(
            embeddings=embeddings,
            corrector=self.corrector,
            num_steps=num_steps,
        )