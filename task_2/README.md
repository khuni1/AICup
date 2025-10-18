# Prerequisite:

Install PyTorch, PyTorch Geometric, Gensim, Scikit-learn

We used: 
-   PyTorch: pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
-   PyTorch Geometric: pip install torch_geometric
-   Gensim: pip install gensim
-   Scikit-learn: pip install scikit-learn

If you faced difficulty in installing, upgrade the pip first: pip install --upgrade pip setuptools

To resolve Numpy Conflict (if)
- Uninstall Numpy: pip uninstall numpy -y
- Install a Compatible Numpy Version: pip install numpy==1.26.4


# About scripts
1. modela.py -> Consists of Trainable Embedding GAT with learnable embeddings + degree (output: ensemble_model_A_trainable_gat.pth)
2. modelb.py -> Consists of Tuned Node2Vec GAT with Fixed Node2Vec Embeddings + Degrees (heads = 4; dropout = 6 + node_embeddings_p.12_q1.0) (output: ensemble_model_B_fixed_gat.pth)
3. modelc.py -> Consist of Simpler GCN method (output: ensemble_model_C_simple_gcn.pth)
4. ensemble.py -> Perform final prediction on unseen test.data (output: final_ensemble_predictions.csv)
4. ensemble_kfold.py -> Perform model evaluation/validation on the training data (output: average accuracy and f1-score across the K-Folds)


# Execution Steps:
1. Run all the model .py (not necessarily to follow in sequence)
2. Run ensemble.py
2. Lastly, run the ensemble_kfold.py


------ ADDITIONAL INFORMATION
## Explanation on the choosen methods 
Before finalizing our implementations, we been testing on few methods (focusing on the feature engineering) to improve the classification result. Then we decided to focus on GNN (due to simpler methods did not improved on the result), that was when the Model C implementation started. 

### Model C
As you already aware, GNN is not only good at classifying common malware types but also effective at identifying the less frequent ones. To feature engineering with GNN, we improve the Node2Vec embeddings method which this model used;
- Enriched node features: combined the node2vec embeddings with local graph features. For each function (node), the feature vector was a concatenation of:
    - node2vec embedding
    - node's in-degree (number of functions that call it)
    - node's out-degree (number of functions it calls) which provided GNN with a richer representation, giving it both global structural context and local connectivity importance.
- We also used a slightly more complex GNN architecture with two GCNConv Layers, which was better suited to process the enriched features and learn more intricate patterns. 

Evaluation on Validation Set - Accuracy: 0.9786; Macro F1 Score: 0.9827

Although the result improved from the baseline, we still not satisfied so here come the second method..

### Model B
As mentioned previously, Node2Vec is implemeted, but to improve, we try to implement GAT (Graph Attention Network) to improve the existing GNN model. GAT is more sophisticated GNN architecture that can dynamically weigh the importance of node's neighbords, which ideal for FCG.  

The key difference from the GCNConv that was implemented on previous model (Model C) is that, a GAT:
- Learns an "attention score" to determine how much each neighbor contributes to the central node's new representation.
- This allows the model to identify and prioritize the most important functions in a call graph.

However, for efficient training with GAT, there is a need to perform a systematic hyperparameter search on the GAT - since while GAT is powerful, their effectiveness depends heavily on parameters like the number of attention heads and dropout rates, which control how the model learns to prioritize neighbor information. 

So we performed a grid search on the heads and dropouts parameters and we founded that the result best at heads = 4; dropout = 0.2 or 0.6 (Accuracy: 0.9821; Macro F1: 0.9856)

Based on this grid search, we figured: 
- Impact of heads: increasing the number of attention from 2 to 4 was beneficial; as model can learn more complex patterns. however, increasing it further to 8, can cause the model’s performance to drop, especially at higher dropout rates. showing sign of the model became too complex and started to overfit the training data. 
- Impact of dropout: dropout is a regularization technique that prevents overfitting. for heads =4, both a low dropout (0.2) and high dropout (0.6) performed well, while the intermediate value (0.4) was less effective. This suggests that GAT is sensitive to the amount of regularization and requires fine-tuning. 

From here, we have identified the optimal hyperparameters for our GAT model. It slightly outperforms Model C. 

Next, we tried to fine tune on the p and q parameters of node2vec to improve the foundational embeddings for our GAT model. Each of the search grid values are stored in the designated file (we only provided node_embeddings_p1.2_q1.0.kv in this submission file since it provided us the highest classification result upon search grid values of p and q parameters)

(Accuracy: 0.9814; Macro F1: 0.9851)

Then we use these configurations to run k-fold validation,  by running the model five separate times (k=5) and averaging the results, the k-fold process smooths out the randomness from SGD, shuffling and dropout.

Final Cross-Validation result for this Model B turned out:
- Average Accuracy: 0.9041 +/- 0.0680 --> showing instability means the model is extremely unstable and highly sensitive to data it is trained on. From here, we figured that the model's ability to generalize is inconsistent.

Due to this, we almost certain that overfitting the training batches combined with insufficient training time are the causes of this instability. 

As our current k-fold scrip tuned for lr= 0.005 for 100 epochs, we tried to run on lr = 0.01 for epochs 200 but still showed extreme instability. Then we rechecked on our current model:
- Architecture: 3-layer GAT (high complexity)
- Batch size: 32 (very small, probably causing noisy gradients)
- Learning rate: 0.01 (too high for small batch size)

Then we ran with lr = 0.001, epochs = 300 and shows stability! But the classification result is not as high as we aim for 99%. 

So we came across new idea for new model to test. While the current combined feature (node2vec + degrees) are good, the GNN still relying on a fixed, precomputed feature set. So we maybe can let the GNN dynamically learn its own features? → trainable embedding layer(?)

### Model A
In this model, we trying to create task-specific, fully optimized features that are learned simultenously with the GAT architecture. For this implementation, we require modification on both data preparation pipeline and also GNN model architecture:
- Data Processing: Instead of generating a node feature tensor from the .kv file, the parser now:
    - Create a global dictionary mapping every unique function name to a unique integer ID (starting from 0). This ID is the lookup index for the embedding layer.
    - data.x will now be a tensor of these integer IDs.
    - The degrees (in-degree and out-degree) will be stored in a separate tensor, data.degrees, for later concatenation. 
- Model architecture: The new GAT_Trainable class will include an nn.Embedding layer to perform the lookup and learn the optimal feature vectors during training. The degrees with be concatenated right before the first GATconv layer. 

During k-fold validation, we figured:
Average Accuracy: 0.9613 +/- 0.0202
Average Macro F1: 0.9606 +/- 0.0289'

--> the result is pretty strong and the best part is, it learned the embeddings itself!

### Ensembling
So we have identified the top three best models that we did so far. To leverage on each model's advantages, we tried to perform ensembling method. 

Model A -> Learns and optimizes function features from scratch, providing the most task-specific prediction.
Model B -> Uses the GAT's attention mechanism on the most stable, pre-calculated topological features.
Model C -> Uses the simpler GCN to provide stability and a non-attentional perspective, correcting GAT errors. 

In our context, it refers to Soft Voting, which is the technique of averaging the output probabilities of multiple, individually trained models to arrive at a single, final prediction. 

This Soft Voting Ensemble Method's purpose is to reduce the variance and bias present in any single model. When the ensemble runs on the unseen test.data, it follows three steps to achieve a consensus prediction:
1. Individual Prediction (Logits): Each of the three trained models runs the forward pass on the same test sample, generating raw prediction scores (logits) for each malware class --> each model analyzes the sample through its own lens (e.g., GAT uses attentionl GCN uses simple averaging)
2. Softmax Averaging: The logits from all three models are converted into probabilities (using the Softmax function). These three probability vectors are then summed and averaged. --> This is the "soft" part of the vote. It weights the prediction by the model's confidence. For example, if Model A is 90% sure and Models B and C are 40% sure, the ensemble prediction will lean toward Model A.
3. Final Classification: The class with the highest average probability is selected as the final, ensembled prediction. --> By relying on the consensus of diverse experts, the ensemble cancels out the small, uncorrelated errors made by single models, resulting in the high final accuracy of 0.99