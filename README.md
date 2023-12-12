# Final Project

## Installation

### Prerequisite

- Cuda (This project is coded in an environment with cuda 11.8)
- Python
    - version: 3.10
- Anaconda ( \[optional\]  \[Python environment management\] )
    - Install anaconda: 
      - Download: https://www.anaconda.com/download
      - Docs for installation: https://docs.anaconda.com/free/anaconda/install/
  - Enable anaconda commands
      - Linux: Done automatically after installation
      - Windows: Find and open installed `Anaconda Prompt`
  - Create environment
      - `conda create -n graphics python=3.10` and follow the prompt
  - Activate environment
      - `conda activate graphics`
- PyTorch
    - Current version: 2.1.1
    - Install:
        - Link: https://pytorch.org/get-started/locally/
        - In this link, select the preferences and run the command provided in the table
          ![img_1.png](DocAssets/img.png)
        - *The command will be different if PyTorch is installed without conda. Make sure to use exact command generated
          by the website*

### Project Install
- Download project
  - `git clone https://github.com/YY-OhioU/GraphicsFinalProject.git`
- Install dependencies:
  - `pip install -r requirements`
- Decompress neural network model
  - Compressed model is under project root
  - Find `model.zip` then unzip it. Place extracted file `model.pth` under `models` folder
  - `cd models && unzip model.zip`

### Run project:
`python main.py`