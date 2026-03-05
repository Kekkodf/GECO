from src.GECO import GECOvec2text
from src.utils import mylogger
import torch
import ir_datasets
import pandas as pd
import os

logger = mylogger.create('test')

dataset = ir_datasets.load("msmarco-passage/trec-dl-2020/judged")
df = pd.DataFrame(dataset.queries_iter())

if __name__ == "__main__":

    geco_model = GECOvec2text(logger=logger)
    mechs = ['laplace', 'gaussian']
    for mech in mechs:
        geco_model.set_mechanism(mech)
        for eps in [1, 10, 50, 1000]:
            geco_model.set_epsilon(eps)
            logger.info(f"GECO model initialized with epsilon={eps} and mechanism={mech}.")
            # make a fake tensor using torch
            input_tensor, noise = geco_model.get_gtr_embeddings(list(df.text))  # Assuming the encoder produces 768-dimensional embeddings
            input_data = input_tensor.cpu()
            obfuscated_data = geco_model.reconstruct(input_data, num_steps=20)
            logger.info(f"Obfuscated data for epsilon={eps} and mechanism={mech}.")
            df['obfuscatedText'] = obfuscated_data
            norms = []
            sims = []
            for original, obfuscated in zip(df.text, df.obfuscatedText):
                original_tensor, _ = geco_model.get_gtr_embeddings([original])
                obfuscated_tensor, _ = geco_model.get_gtr_embeddings([obfuscated])
                norm = torch.norm(original_tensor - obfuscated_tensor)
                sim = torch.nn.functional.cosine_similarity(original_tensor, obfuscated_tensor)
                norms.append(norm.item())
                sims.append(sim.item())
            df['Embedding_distance'] = norms
            df['Cosine_similarity'] = sims
            #save in test/results/ with the name geco_results_eps_{eps}.csv
            #verify that the directory exists, if not create it
            if not os.path.exists('test/results'):
                os.makedirs('test/results')
            df.to_csv(f'test/results/obfuscatedTexts_GECOVec2Text_{mech}_{eps}.csv', index=False)
            logger.info(f"Results saved to test/results/obfuscatedTexts_GECOVec2Text_{mech}_{eps}.csv")
            #logger.info("Input string: {}".format("do goldfish grow"))
            #reconstructed_data = geco_model.reconstruct(input_data, num_steps=5)
            #logger.info("Reconstructed data: {}".format(reconstructed_data[0]))
            #logger.info("Noise: {}".format(noise))
            #output_tensor, noise = geco_model.get_gtr_embeddings(["do goldfish grow"])
            #norm = torch.norm(input_tensor - output_tensor)
            #logger.info('L2 norm between input and output embeddings: {}'.format(norm.item()))
            #sim = torch.nn.functional.cosine_similarity(input_tensor, output_tensor)
            #logger.info('Cosine similarity between input and output embeddings: {}'.format(sim.item()))