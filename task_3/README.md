# Prerequisite:
Install PyTorch, Hugging Face Transformers, and essential libraries for machine learning and data processing.

We used:
-   PyTorch (Nightly, CUDA 12.1):  
	pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

-   Hugging Face Transformers (v4.57.0):  
	pip install git+https://github.com/huggingface/transformers

-   Default Libraries:  
	pip install numpy>=1.23.0 pandas>=1.5.0 matplotlib>=3.7.0

-   Machine Learning:  
	pip install scikit-learn>=1.2.0 datasets

-   Additional Utilities:  
	pip install tqdm>=4.65.0 sentencepiece protobuf tiktoken tokenizers accelerate safetensors pillow torchvision peft

If you faced difficulty in installing, upgrade pip first:  
	pip install --upgrade pip setuptools

To resolve Numpy Conflict (if):  
-   Uninstall Numpy: pip uninstall numpy -y  
-   Install a Compatible Numpy Version: pip install numpy==1.26.4

# About scripts
-	nomalize_dataset_answer.py → Reads a JSONL file and normalizes the answer field by removing units and leading comparison symbols (supports scientific notation such as ×10^k), while keeping non-numeric labels intact (e.g., “B”, “无法确定”, “Fe(OH)3”). It writes the cleaned results line-by-line to nomalized_train.json and prints normalization statistics. (Note: Since a normalized file already exists in the directory, this script does not need to be executed again unless you want to regenerate it.)

-   train.py → Performs leak-free K-fold fine-tuning of Qwen2.5-VL-7B for photographed Chinese table QA using a chat template and optional solution supervision. Supports LoRA and 8-bit optimizers, evaluates an inner calibration split at each checkpoint, and conducts an external OOF evaluation—saving fold checkpoints, processors, per-fold cal/val predictions, and an OOF accuracy summary (output: qwen25vl_kfold/fold-*/, qwen25vl_kfold/oof_predictions.csv, qwen25vl_kfold/oof_summary.json).

-   test.py → Loads the Qwen2.5-VL-7B fold-2 checkpoint (with PEFT fallback) and processor, builds inference prompts for photographed Chinese-table QA, runs deterministic generation per item (from test.json + images), normalizes answers, and saves predictions to CSV (output: fold2_test_predictions.csv).

-	fold2_test_predictions.csv → Since LoRA fine-tuning and inference with Qwen2.5 require a long runtime, we provide the prediction results on the test dataset generated using the LoRA fine-tuned model from fold 2.


# Execution Steps:
1. cd task_3/machine_learning/qwen25vl
2. Run train.py
3. Run test.py

