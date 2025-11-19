## Third-Party Dependencies

- `OSTrack/`: Official single-object tracking implementation kept outside version control to avoid pushing a large upstream repository.
- `DiffuEraser/`: Diffusion-based video inpainting repo ([upstream](https://github.com/lixiaowen-xw/DiffuEraser)) invoked via our wrapper.

Recommended workflow:
1. Clone the upstream projects into `third_party/OSTrack` and `third_party/DiffuEraser`.
2. Follow each README to download pretrained weights (OSTrack checkpoints, Stable Diffusion 1.5, VAE, DiffuEraser weights, Propainter weights, etc.).
3. Keep custom changes as patches or forks rather than committing vendored code/weights into this repository.