# ESM-AA Model Official Repository

Welcome to the official repository for the ESM-AA (ESM-All Atom) model. This repository contains the source code and usage instructions for the ESM-AA model, developed based on the official ESM codes by FAIR. You can find the original ESM repository [here](https://github.com/facebookresearch/esm).

## Environment Setup

To set up the necessary environment for using the ESM-AA model, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/zhengkangjie/ESM-AA.git
   cd ESM-AA
   ```

2. **Create and activate the conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate esmfold
   ```

3. **Install the package**:
   ```bash
   python setup.py install
   ```

## Model Checkpoints

We have made available checkpoints for the ESM-AA model. There is one version available now and we will release more checkpoints of ESM-AA with larger scale soon:

- **ESM-AA 35M with Regression Weight**: Download from [this link](https://drive.google.com/drive/folders/13VjjFJ8Ger5zwrEs0iafe7p69y6K1lBo?usp=sharing).



## Example Usage

Currently, we provide an example for the unsupervised contact prediction task using the same settings as the original ESM model. You can find the example code in the `examples` directory.

Here is an example to load ESM-AA:

```python
import esm

# Load model and alphabet
esm_aa, esm_aa_alphabet = esm.pretrained.esm_aa_t12_35M('path/to/checkpoint')
```

## Future Releases

We plan to release code for protein-molecular tasks soon. Stay tuned for updates and more functionalities!

## Support

For questions and support, please open an issue in the GitHub repository or contact us directly through our provided communication channels.
