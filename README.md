# BERT-CLS-MLC
Multi-class multi-label classification with BERT CLS token representation.

## 💾Dataset

SemEval-2018 Task1 E-c-English. See the [official website](https://competitions.codalab.org/competitions/17751).

## ⚙️Dependencies

- python == 3.7.7
  - torch == 1.5.1
  - transformers == 3.0.2
  - numpy == 1.19.4
  - tqdm == 4.53.0
  - sklearn == 0.21.3
  - deepspeed == 0.4.1

## 🚀QuickStart

```python
python run.py
```

## 🔬Experiments

- GPU: Titan Xp * 1 (12GB)
- CPU: Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz
- Batch size: 64

| Name                          | Epoch | Max GPU Memory (MB) | Train Time/epoch (s) | Test Time/epoch (s) | Macro F1 (%) |
| :---------------------------- | ----: | ------------------: | -------------------: | ------------------: | -----------: |
| Baseline                      |     3 |                8835 |                   26 |                   3 |       46.632 |
| DeepSpeed (stage 0)           |     3 |                5647 |                   26 |                   3 |       46.632 |
| DeepSpeed (stage 1)           |     3 |                5757 |                   29 |                   3 |       46.155 |
| DeepSpeed (stage 2)           |     3 |                5757 |                   30 |                   3 |       46.155 |
| DeepSpeed (stage 3, disabled) |     3 |                7143 |                   51 |                   8 |       46.491 |

## 🎫Licence

This project is under MIT licence. For licence of dataset, refer [this](data/README.txt) for more information.
