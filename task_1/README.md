# Prerequisite:
Install PyTorch, Hugging Face Transformers, and essential libraries for machine learning and data processing.

We used:
-   PyTorch (Nightly, CUDA 12.1):  
	pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

-   Hugging Face Transformers and Datasets:  
	pip install transformers datasets>=2.12.0

-   Default Libraries:  
	pip install numpy>=1.23.0 pandas>=1.5.0 matplotlib>=3.7.0

-   Machine Learning:  
	pip install scikit-learn>=1.2.0

-   Additional Utilities:  
	pip install tqdm>=4.65.0 sentencepiece protobuf tiktoken tokenizers accelerate safetensors joblib

If you faced difficulty in installing, upgrade pip first:  
	pip install --upgrade pip setuptools

To resolve Numpy Conflict (if):  
-   Uninstall Numpy: pip uninstall numpy -y  
-   Install a Compatible Numpy Version: pip install numpy==1.26.4

# About scripts
-   train.py → Performs 5-fold document-level classification using BigBird-RoBERTa with Platt/Isotonic calibration, saving per-fold models, calibrated OOF metrics, confusion matrices, and FP/FN reports (output: final_model/ with calibrated thresholds and fold models).
-   test.py → Loads BigBird-RoBERTa fold models and Platt/Isotonic calibrators, tokenizes test .tex docs, produces per-fold probabilities and ensembles them (mean/logit_mean/topk3) with θ=0.5, saving final_model_result/test_predictions.csv plus distribution plots and JSON/TXT summaries.


# Execution Steps:
1. cd task1/machine_learning/bigbird
2. Run train.py
3. Run test.py

