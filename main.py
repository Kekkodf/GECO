import geco.utils

from geco.src.AbstractGECO import Obfuscator
from geco.src.KernelDensityGECO import KernelDensityGECO
from geco.src.CosineGECO import CosineGECO
from geco.src.CorrelationGECO import CorrelationGECO

DEMO_TEXTS = [
    "do goldfish grow",
    "what is wifi vs bluetooth",
    "why did the us volunterilay enter ww1",
    "definition declaratory judgment",
    "right pelvic pain causes"
]
EPSILON_VALUES = [1.0, 5.0, 10.0, 12.5, 15.0, 17.5, 20.0, 30.0, 50.0]

def main():
    log = geco.utils.createLogger('main')
    #log.warning("This is a demo of the GECO obfuscator. The obfuscated outputs may not be perfect and may contain errors. Please use with caution and do not rely on the obfuscated outputs for any critical tasks.")
    st_model_name = "sentence-transformers/gtr-t5-base"
    inversion_model_name = "ielabgroup/vec2text_gtr-base-st_inversion"#jxm/gtr__nq__32,ielabgroup/vec2text_gtr-base-st_inversion
    corrector_model_name = "ielabgroup/vec2text_gtr-base-st_corrector"#jxm/gtr__nq__32__correct,ielabgroup/vec2text_gtr-base-st_corrector
    #geco_obfuscator = geco.src.AbstractGECO.Obfuscator(
    #    log=log,
    #    st_model_name=st_model_name,
    #    inversion_model_name=inversion_model,
    #    corrector_model_name=corrector_model,
    #    cache_path_pins="./data/query_pool/msmarco_gtr_embeddings.pt",
    #    collection_name="msmarco-passage/train",
    #    top_percentile=0.5,
    #    encode_batch_size=128
    #)
    #print(geco_obfuscator)
    #geco_obfuscator.check_utility_function() # Check utility function for debugging and repro. This will log the utility function being used in the obfuscator.
    
    geco_obfuscator_test2 = Obfuscator(
        log=log,
        st_model_name=st_model_name,
        inversion_model_name=inversion_model_name,
        corrector_model_name=corrector_model_name,
        cache_path_pins="./data/query_pool/msmarco_gtr_embeddings.pt",
        collection_name="msmarco-passage/train",
        top_percentile=0.5,
        encode_batch_size=128
    )
    
    exit()
    
    print("\n" + "=" * 76)
    for text in DEMO_TEXTS:
      for eps in EPSILON_VALUES:
        print(geco_obfuscator.obfuscate(text, epsilon=eps))

if __name__ == "__main__":
    main()
    
    