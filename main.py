from src.GECO import GECOvec2text
from src.utils import mylogger
import torch


logger = mylogger.create('test')

if __name__ == "__main__":
    geco_model = GECOvec2text(logger=logger)
    logger.info("GECO model initialized.")
    # make a fake tensor using torch
    input_tensor, noise = geco_model.get_gtr_embeddings(["do goldfish grow"])  # Assuming the encoder produces 768-dimensional embeddings
    input_data = input_tensor.cpu()
    logger.info("Input string: {}".format("do goldfish grow"))
    reconstructed_data = geco_model.reconstruct(input_data, num_steps=5)
    logger.info("Reconstructed data: {}".format(reconstructed_data[0]))
    logger.info("Noise: {}".format(noise))
    output_tensor, noise = geco_model.get_gtr_embeddings(["do goldfish grow"])
    norm = torch.norm(input_tensor - output_tensor)
    logger.info('L2 norm between input and output embeddings: {}'.format(norm.item()))
    sim = torch.nn.functional.cosine_similarity(input_tensor, output_tensor)
    logger.info('Cosine similarity between input and output embeddings: {}'.format(sim.item()))