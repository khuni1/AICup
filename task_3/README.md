### Task 3 Setup & Execution

1. **It is recommended to install Python 3.10.14.**
    
    This project **requires CUDA** for GPU acceleration.
    
    Please ensure that CUDA (version 12.1 or compatible) is properly installed on your system.
    
    Then, change to the following directory:
    
    ```bash
    cd machine_learning/qwen25vl
    ```
    
2. **Install required packages**
    
    ```bash
    pip install -r requirements.txt
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
    ```
    
3. **Training data nomalizing** 
    
    The **training dataset** used in this project is the normalized version, `normalized_train.json`.
    
    The normalization process, implemented in `/train/normalize_dataset_answer.py`, removes units from numerical answers. If the `answer` field contains a number (including decimals or scientific notation) followed by a unit, the unit is stripped.
    
    Additionally, when comparison symbols such as ≥, ≤, or ≈ precede the value, only the numeric part is retained, while non-numeric answers (e.g., categorical labels or chemical formulas) remain unchanged.
    
4. **Training**
    
    ```bash
    python3 train.py
    ```
    
    - When you run `train.py`, the model will perform **k-fold LoRA training**.
    - Please note that **LoRA tuning can take a significant amount of time**.
    
    During this process: 
        Each fold’s evaluation results are stored in the `qwen25vl_kfold` directory.

5. **Test**
    
    ```bash
    python3 test.py
    ```
    
    - Running `test.py` loads the trained checkpoint from a specific fold (**default = fold-2**) and performs inference on the test dataset.
    - The inference results are saved as **`foldn_test_predictions.csv`**, where `n` indicates the fold number (e.g., `fold2_test_predictions.csv`).