import argparse
import os
import pandas as pd
import ir_datasets

from geco.utils import createLogger
from geco.src.AbstractGECO import Obfuscator
from geco.src.CorrelationGECO import CorrelationGECO
from geco.src.CosineGECO import CosineGECO
from geco.src.KernelDensityGECO import KernelDensityGECO
from geco.src.ExponentialGECO import ExponentialGECO

DEMO_TEXTS = [
    "do goldfish grow",
    "what is wifi vs bluetooth",
    "why did the us volunterilay enter ww1",
    "definition declaratory judgment",
    "right pelvic pain causes"
]

def main():
    log = createLogger('main')
    
    geco_obfuscator= KernelDensityGECO(
        log=log, strategy="internal"
    )
    print(geco_obfuscator)
    

    print("\n" + "=" * 76)
    for text in DEMO_TEXTS:
        geco_obfuscator.obfuscate(text, epsilon=1.0)
        

if __name__ == "__main__":
    main()
    
    