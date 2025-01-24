
####  Code1: Transformer Model ####
=================================== 1. Overview ====================================
	This repository contains the code to implement a Transformer model for orientation coding in macaque V1 neurons. The model is designed to reconstruct Gabor image based on neuronal responses. We take ‘Demo’ as example. Demo contains 150 neurons.
 
	This repository includes the following components:
		Data preprocessing
		Transformer model structure
		Training and testing

=================================== 2. Requirements ====================================
	Hardware:
		NVIDIA GPU with at least 24GB of memory.
		A standard computer with enough RAM.
	Software:
		Python & Matlab (Codes are tested on version 2022b)

	The following dependencies are required to run the code:
		Python 3.8 (recommended version: 3.8)
		Pycharm 2023.2.3

	Setting up the Environment in PyCharm:
	If you're using PyCharm, follow these steps to set up your Python environment and install the necessary dependencies:
	Open PyCharm and navigate to PyCharm > Preferences (on macOS) or File > Settings (on Windows/Linux).
	In the Preferences/Settings window, go to Project: transformer_model > Python Interpreter.
	Select Python 3.8 as the interpreter:
		If Python 3.8 is already selected (like in your case), you can click on the Python 3.8 dropdown to verify or add a new interpreter.
		If Python 3.8 is not listed, click Add Interpreter to add the desired Python version.

	Install dependencies:
		Once the correct interpreter is selected, click the + icon in the Python Packages section to install the required packages.
		Search for and install packages:
			torch==1.7.1+cu110
			hdf5storage==0.1.19
			numpy==1.24.4
			matplotlib==3.7.3
			scikit-learn==1.4.1
			scipy==1.10.1

		Alternatively, you can use the PyCharm Terminal to install dependencies. In the terminal (inside PyCharm), run:
			pip install torch==1.7.1+cu110 numpy matplotlib …

	The entire installation and setup process should take approximately 20 minutes.

===================================3.File structure====================================
The project directory is organized as follows:

Code/
├── __pycache__/              # Python cache files
├── Demo/                       # Metadata
│   └── Data/      
│         ├── G4_RespAvg.mat   # Responses: [144 * neuron_number(150)]
│         └── order_fig_1st_100.mat	# Stimuli image: [144 * 79 * 79]
├── Model/                    # Network model components
│   ├── AttentionWithNoDeconvPos.py  # The overall architecture is based on the Transformer model
│   ├── DataPre.py            # Data preprocessing script
│   ├── Embedding.py          # Embedding layers for transformer
│   ├── ModelTester.py        # Script for testing the model
│   ├── ModelTrainer.py       # Script for training the model
│   ├── multi_head_attention.py  # Multi-head attention mechanism
│   ├── positional_encoding.py    # Positional encoding layer
│   ├── Run.py                # Script to execute the model pipeline
│   ├── scale_dot_product_attention.py  # Scaled dot product attention
│   └── unEmbedding.py        # Unembedding layers for transformer

===================================4. Usage====================================
	To run the complete pipeline, including training and testing, simply execute the Run.py script. Simply run Run.py by clicking the Run button or using the Run > Run 'Run' menu in PyCharm.
	This script will automatically:
		Train the model by calling the ModelTrainer.py and test the model after training by calling the ModelTester.py.
		After training, the trained model will be saved to the Model folder.
		Save the associated analysis data in Demo

If the batch size is 2, epochs are 1000, and the number of neurons is 150, the total training and testing time for the entire program is 714.59 seconds.


Analysis data was stored in Demo folder:
 Demo/ 
 ├── loss_data.mat  # contains the train and validation loss
 ├── test_data.mat  # contains the R2 values for each trial and the mean R2 value
 ├── attention_map.mat  # the core of the transformer model, the attention map for each trial, dimension: [trials * neuron * neuron], trials = 144
 ├── output_true.mat  #  original stimuli images [trials * 79 * 79]
 └── output_pred.mat  # predicted stimuli images [trials * 79 * 79]
  
####  Code2: Analysis ####

=============================== 1. Overview ==============================
In the analysis phase, the attention map is progressively masked based on a thresholding. The following process is applied to the attention map:
Column Mean Calculation: First, the mean of each column in the attention map is calculated.
Thresholding: A series of thresholds are applied to selectively mask parts of the attention map.  settings: thresholds = 10 ** np.arange(-1, -3.2, -0.05)
Masking: Based on the threshold, the columns with values greater than (or less than) the threshold are kept, while the others are masked (set to zero).

=============================== 2. File structure ==============================

Model/
├── __pycache__/                 # Compiled Python files (.pyc)
│   └── <compiled_files>.pyc
├── Demo_cluster_model.pth         # Pre-trained model (PyTorch format, GPU)
├── AttentionWithNoDeconvPos.py  
├── DataPre.py                   
├── Embedding.py                
├── multi_head_attention.py    
├── positional_encoding.py      
├── scale_dot_product_attention.py # in addition to implementing the standard Scaled Dot-Product Attention mechanism, it also includes the progressive masking process
├── try.py                       # main
└── unEmbedding.py         


=============================== 3. Usage ==============================
To run the try.py file (Simply run try.py by clicking the Run button) and generate the results.mat file in the Demo folder. This allows for observation of how the attention map changes and how different thresholds impact the model's results.

After the masking process and evaluation, the results.mat file is generated. This file contains the reconstructed stimulus images obtained by applying different thresholds （the mean of each column） to the attention map.  This file was stored in Demo. The dimensions of results.mat are as follows:
	44 steps: Each step corresponds to a different threshold.
	2 cases: 1 represents keeping the connections of key neurons, and 2 represents discarding the connections of key neurons.
	12 orientations: Different orientations of the stimulus.
	79x79: The spatial resolution of the reconstructed images.

The analysis runtime is 12.99 seconds.



