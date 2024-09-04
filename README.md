# SC-VAE
This repository contains official implementation for the paper titled "SC-VAE: Sparse Coding-based Variational Autoencoder with Learned ISTA".

# Installing Dependencies
To install dependencies, create a conda or virtual environment with Python 3 and then run `pip install -r requirements.txt`.

# Running the SC-VAE
To run the SC-VAE simply run `python main-stage1.py`. You could change the config files in `line 279` to train SC-VAE model with different downsampling blocks.
```python
parser.add_argument('--model-config', type=str, default='./configs/ffhq/stage1/ffhq256-scvae16x16.yaml')
```

# Citation
@article{xiao2023sc,    
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;title={SC-VAE: Sparse Coding-based Variational Autoencoder with Learned ISTA},    
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;author={Xiao, Pan and Qiu, Peijie and Ha, Sung Min and Bani, Abdalla and Zhou, Shuang and Sotiras, Aristeidis},    
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;journal={Available at SSRN 4794775},    
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;year={2023}    
}

# To-Do List
- [x] Installing Dependencies
- [ ] Training the Model
- [ ] Evaluating the Model
- [ ] Upload Pre-trained SC-VAEs



