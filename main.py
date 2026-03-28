from geco.utils import mylogger
import geco.src.GECO as geco
import vec2text

DEMO_TEXTS = [
    "do goldfish grow",
    "what is wifi vs bluetooth",
    "why did the us volunterilay enter ww1",
    "definition declaratory judgment",
    "right pelvic pain causes"
]
EPSILON_VALUES = [1.0, 5.0, 10.0, 12.5, 15.0, 17.5, 20.0, 30.0, 50.0]

def main():
    log = mylogger.create('main')
    inversion_model = "ielabgroup/vec2text_gtr-base-st_inversion"#jxm/gtr__nq__32,ielabgroup/vec2text_gtr-base-st_inversion

    log.info("Loading vec2text corrector model ...")
    corrector_model = "ielabgroup/vec2text_gtr-base-st_corrector"#jxm/gtr__nq__32__correct,ielabgroup/vec2text_gtr-base-st_corrector
    geco_obfuscator = geco.GECO(log, 
                                st_model="all-MiniLM-L6-v2",
                                inversion_model=inversion_model,
                                corrector_model=corrector_model,
                                )
    
    print("\n" + "=" * 76)
    for text in DEMO_TEXTS:
      for eps in EPSILON_VALUES:
        print(geco_obfuscator.obfuscate(text, epsilon=eps))

if __name__ == "__main__":
    main()
    
    