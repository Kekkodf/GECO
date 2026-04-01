import geco.utils
from geco.src.AbstractGECO import Obfuscator
from geco.src.CorrelationGECO import CorrelationGECO
from geco.src.CosineGECO import CosineGECO
from geco.src.KernelDensityGECO import KernelDensityGECO

DEMO_TEXTS = [
    "do goldfish grow",
    "what is wifi vs bluetooth",
    "why did the us volunterilay enter ww1",
    "definition declaratory judgment",
    "right pelvic pain causes"
]

def main():
    log = geco.utils.createLogger('main')
    st_model_name = "sentence-transformers/gtr-t5-base"
    inversion_model_name = "ielabgroup/vec2text_gtr-base-st_inversion"#jxm/gtr__nq__32,ielabgroup/vec2text_gtr-base-st_inversion
    corrector_model_name = "ielabgroup/vec2text_gtr-base-st_corrector"#jxm/gtr__nq__32__correct,ielabgroup/vec2text_gtr-base-st_corrector
    
    geco_obfuscator_test2 = KernelDensityGECO(
        log=log,
        st_model_name=st_model_name,
        inversion_model_name=inversion_model_name,
        corrector_model_name=corrector_model_name,
        cache_path_pins="./data/query_pool/msmarco_gtr_embeddings.pt",
        collection_name="msmarco-passage/train",
        top_percentile=0.001
    )
    print(geco_obfuscator_test2)
    
    
    print("\n" + "=" * 76)
    for text in DEMO_TEXTS:
      for eps in geco_obfuscator_test2.epsilons:
        print(geco_obfuscator_test2.obfuscate(text, epsilon=eps))

if __name__ == "__main__":
    main()
    
    