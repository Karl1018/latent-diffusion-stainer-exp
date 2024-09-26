import torch
from omegaconf import OmegaConf
from sklearn.manifold import TSNE

from ldm.data.bci import IHCDataset
from ldm.util import instantiate_from_config

def sample_ae():
    ckpt_path = "/home/karl/latent-diffusion/logs/2024-09-11T17-30-40_autoencoder_kl_bci/checkpoints/epoch=000988.ckpt"
    config_path = "/home/karl/latent-diffusion/configs/autoencoder/autoencoder_kl_bci.yaml"

    config = OmegaConf.load(config_path)

    ae = instantiate_from_config(config.model)
    ae.init_from_ckpt(ckpt_path)

    # Sample from the model
    z = torch.randn(1, 3, 32, 32)
    sample = ae.decode(z)

    return sample, ae

if __name__ == "__main__":
    sample, ae = sample_ae()
    print(sample.shape)

    # Save the sample as a file
    import torchvision
    torchvision.utils.save_image(sample, "sample.png")

    # Use t-SNE to visualize the latent space
    x = torch.randn(50, 3, 128, 128)
    z = ae.encode(x) # x is DiagonalGaussianDistribution
    sample = z.sample().detach().cpu().numpy()
    # Reshape the sample
    sample = sample.reshape(sample.shape[0], -1)

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=20)
    latents_tsne = tsne.fit_transform(sample)

    # Plot the t-SNE
    import matplotlib.pyplot as plt
    plt.scatter(latents_tsne[:, 0], latents_tsne[:, 1])
    plt.show()
