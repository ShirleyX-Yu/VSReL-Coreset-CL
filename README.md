# Coreset Selection via Reducible Loss in Continual Learning

A natural solution for rehearsal-based continual learning is to select a coreset as memory. A coreset serves as an informative summary of a large dataset, enabling a model trained solely on the coreset to achieve performance comparable to training on the full dataset. Previous bi-level coreset selection methods adjust sample weights or probabilities to minimize the outer loss, which is computed over the entire dataset. For non-representative samples like ambiguous or noisy samples, since these samples are not well learned even training model on the full dataset, loss of these samples in the outer loss are not worthy to be reduced. However, their high loss values may cause them to be selected in an attempt to minimize the outer loss, which may lead to suboptimal performance for models trained on the coreset. To address this issue, we first investigate how the performance of a trained model changes when a sample is added to the training dataset and approximate this performance gain using reducible loss. We then select samples with the highest performance gain in the coreset so that performance of model trained on coreset could be maximized. We show that samples with high performance gain are informative and representative. Furthermore, reducible loss requires only forward computation, making it significantly more efficient than previous methods. To better apply coreset selection in continual learning, we extend our method to address key challenges such as task interference, streaming data, and knowledge distillation. Experiments on data summarization and continual learning demonstrate the effectiveness and efficiency of our approach.

## Run experiments

**Note:** code and scripts will be uploaded later.

## Citation

If you find our work helpful, please cite our paper by the following reference:

```
@inproceedings{author2025title,
  title={Coreset Selection via Reducible Loss in Continual Learning},
  author={Ruilin Tong, Yuhang Liu, Javen Qinfeng Shi, Dong Gong},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2025},
  url={https://openreview.net/forum?id=mAztx8QO3B},
}
```
