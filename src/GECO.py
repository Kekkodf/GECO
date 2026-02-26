import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class GECO:
    def __init__(self, model_name='t5-base', device='cuda'):
        ...