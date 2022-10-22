# iDNA-ABF-automl

This repository is a nni version and base on pytorchlighting to the model [iDNA-ABF](https://github.com/FakeEnd/iDNA_ABF)

We do not use adversarial training in this repository, so the results may be little lower than the metrics in the paper, but they are still higher than the other methods.

## How to use
### main
- train_ABF.py -> train and test model
- train_model.py

### nni
- fusion: nnictl create -p 9990 -c config_idna.yml
- bert: nnictl create -p 9990 -c config_bert.yml
- searh_space_idna.json[fusion]
- searh_space_bert.json[bert]

### module
- data_module: onehot: (1) true -> auto tokenlize (2) false -> input directly
- lightning_module: add models


## The pytorchlighting parameters and results
### 5hmC_H.sapiens (fMkaF)

```json
{
    "batch_size": 64,
    "lr": 0.00005,
    "dropout": 0.7,
    "alpha": [
        0.4,
        0.6
    ]
}
```

| ACC  | AUC  | MCC  | F1   | F2   | F3   |
| ---- | ---- | ---- | ---- | ---- | ---- |
|   0.950512   |   0.971124   |   0.902456   | 0.951867 | 0.967769 | 0.973189 |
| Q    | SE   | SP   | PPV  | NPV  |      |
| 0.950512 | 0.978669 | 0.922355 | 0.926494 | 0.977396 |      |



### 5hmC_M.musculus (fMkaF)

```json
{
    "batch_size": 16,
    "lr": 0.0001,
    "dropout": 0.5,
    "alpha": [
        0.2,
        0.8
    ]
}
```

| ACC  | AUC  | MCC  | F1   | F2   | F3   |
| ---- | ---- | ---- | ---- | ---- | ---- |
|   0.967645   |   0.976875   |   0.935292   | 0.967672 | 0.968145 | 0.968303 |
| Q    | SE   | SP   | PPV  | NPV  |      |
| 0.967645 | 0.968461 | 0.96683 | 0.966884 | 0.96841 |      |



### 4mC_C.equisetifolia (ie9Je)

```json
{
    "batch_size": 128,
    "lr": 0.0005,
    "dropout": 0.7,
    "alpha": [
        0.4,
        0.6
    ]
}
```

| ACC  | AUC  | MCC  | F1   | F2   | F3   |
| ---- | ---- | ---- | ---- | ---- | ---- |
|   0.846995   |   0.902177   |   0.698171   | 0.83815 | 0.810056 | 0.801105 |
| Q    | SE   | SP   | PPV  | NPV  |      |
| 0.846995 | 0.79235 | 0.901639 | 0.889571 | 0.812808 |      |



### 4mC_F.vesca (X8Q6Y)

```json
{
    "batch_size": 32,
    "lr": 0.0001,
    "dropout": 0.4,
    "alpha": [
        0.4,
        0.6
    ]
}
```

| ACC  | AUC  | MCC  | F1   | F2   | F3   |
| ---- | ---- | ---- | ---- | ---- | ---- |
|   0.851291   |   0.923053   |   0.70327   | 0.854506 | 0.865735 | 0.869543 |
| Q    | SE   | SP   | PPV  | NPV  |      |
| 0.851291 | 0.873386 | 0.829197 | 0.836425 | 0.867532 |      |



### 4mC_S.cerevisiae (MRjYH)

```json
{
    "batch_size": 16,
    "lr": 0.00005,
    "dropout": 0.5,
    "alpha": [
        0.5,
        0.5
    ]
}
```

| ACC  | AUC  | MCC  | F1   | F2   | F3   |
| ---- | ---- | ---- | ---- | ---- | ---- |
|   0.720425   |   0.775838   |   0.44199   | 0.710016 | 0.694501 | 0.68948 |
| Q    | SE   | SP   | PPV  | NPV  |      |
| 0.720425 | 0.68453 | 0.75632 | 0.737473 | 0.70566 |      |



###  4mC_Tolypocladium (MUVm1)

```json
{
    "batch_size": 32,
    "lr": 0.00005,
    "dropout": 0.1,
    "alpha": [
        0.5,
        0.5
    ]
}
```

| ACC      | AUC      | MCC     | F1       | F2       | F3      |
| -------- | -------- | ------- | -------- | -------- | ------- |
| 0.737896 | 0.814666 | 0.475839 | 0.736054 | 0.732962 | 0.731937 |
| Q        | SE       | SP      | PPV      | NPV      |         |
| 0.737896 | 0.730915  | 0.744878 | 0.741265 | 0.73462  |         |



###  6mA_A.thaliana (i1D5l)

```json
{
    "batch_size": 64,
    "lr": 0.00005,
    "dropout": 0.5,
    "alpha": [
        0.5,
        0.5
    ]
}
```

| ACC      | AUC      | MCC     | F1       | F2       | F3      |
| -------- | -------- | ------- | -------- | -------- | ------- |
| 0.858622 | 0.93164 | 0.717525 | 0.856616 | 0.849383 | 0.846999 |
| Q        | SE       | SP      | PPV      | NPV      |         |
| 0.858622 | 0.844629  | 0.872615 | 0.868948 | 0.848859  |         |



###  6mA_C.elegans (YUQ7c)

```json
{
    "batch_size": 128,
    "lr": 0.00005,
    "dropout": 0.3,
    "alpha": [
        0.5,
        0.5
    ]
}
```

| ACC      | AUC      | MCC     | F1       | F2       | F3      |
| -------- | -------- | ------- | -------- | -------- | ------- |
| 0.910176 | 0.966053 | 0.820387 | 0.910591 | 0.913126 | 0.913974 |
| Q        | SE       | SP      | PPV      | NPV      |         |
| 0.910176 | 0.914824  | 0.905528 | 0.906398 | 0.914025  |         |



###  6mA_C.equisetifolia (ICjHp)

```json
{
    "batch_size": 64,
    "lr": 0.00005,
    "dropout": 0.7,
    "alpha": [
        0.4,
        0.6
    ]
}
```

| ACC      | AUC      | MCC     | F1       | F2       | F3      |
| -------- | -------- | ------- | -------- | -------- | ------- |
| 0.722717 | 0.803041 | 0.44784 | 0.736364 | 0.75877 | 0.766545 |
| Q        | SE       | SP      | PPV      | NPV      |         |
| 0.722717 | 0.774481  | 0.670953 | 0.906398 | 0.701823  |         |



###  6mA_D.melanogaster (ICjHp)

```json
{
    "batch_size": 128,
    "lr": 0.0001,
    "dropout": 0.3,
    "alpha": [
        0.4,
        0.6
    ]
}
```

| ACC      | AUC      | MCC     | F1       | F2       | F3      |
| -------- | -------- | ------- | -------- | -------- | ------- |
| 0.92109 | 0.969753 | 0.842273 | 0.920501 | 0.916392 | 0.91503 |
| Q        | SE       | SP      | PPV      | NPV      |         |
| 0.92109 | 0.913673  | 0.928508 | 0.927431 | 0.914935  |         |



###  6mA_F.vesca (FlBH6)

```json
{
    "batch_size": 128,
    "lr": 0.00005,
    "dropout": 0.7,
    "alpha": [
        0.4,
        0.6
    ]
}
```

| ACC      | AUC      | MCC     | F1       | F2       | F3      |
| -------- | -------- | ------- | -------- | -------- | ------- |
| 0.941973 | 0.979585 | 0.883982 | 0.942234 | 0.944781 | 0.945633 |
| Q        | SE       | SP      | PPV      | NPV      |         |
| 0.941973 | 0.946486  | 0.93746 | 0.938019 | 0.945999  |         |



###  6mA_H.sapiens (n7zZ9)

```json
{
    "batch_size": 32,
    "lr": 0.0001,
    "dropout": 0.7,
    "alpha": [
        0.5,
        0.5
    ]
}
```

| ACC      | AUC      | MCC     | F1       | F2       | F3      |
| -------- | -------- | ------- | -------- | -------- | ------- |
| 0.905367 | 0.966387 | 0.811073 | 0.903979 | 0.896094 | 0.893496 |
| Q        | SE       | SP      | PPV      | NPV      |         |
| 0.905367 | 0.890913  | 0.919821 | 0.917434 | 0.893978  |         |



###  6mA_R.chinensis (xWuPj)

```json
{
    "batch_size": 16,
    "lr": 0.0001,
    "dropout": 0.7,
    "alpha": [
        0.4,
        0.6
    ]
}
```

| ACC      | AUC      | MCC     | F1       | F2       | F3      |
| -------- | -------- | ------- | -------- | -------- | ------- |
| 0.8628 | 0.94 |  | 0.87 |  |  |
| Q        | SE       | SP      | PPV      | NPV      |         |
|  | 0.88  | 0.85 |  |   |         |

Mention: some problems lead to the interrupt during the training process, this is the result before interrupt.



###  6mA_S.cerevisiae (vImsc)

```json
{
    "batch_size": 64,
    "lr": 0.00005,
    "dropout": 0.5,
    "alpha": [
        0.4,
        0.6
    ]
}
```

| ACC      | AUC      | MCC     | F1       | F2       | F3      |
| -------- | -------- | ------- | -------- | -------- | ------- |
| 0.830164 | 0.902029 | 0.661092 | 0.825981 | 0.813953 | 0.810022 |
| Q        | SE       | SP      | PPV      | NPV      |         |
| 0.830164 | 0.806128  | 0.8542 | 0.846837 | 0.81502  |         |



###  6mA_R.chinensis (rVXQG)

```json
{
    "batch_size": 32,
    "lr": 0.00005,
    "dropout": 0.1,
    "alpha": [
        0.5,
        0.5
    ]
}
```

| ACC      | AUC      | MCC     | F1       | F2       | F3      |
| -------- | -------- | ------- | -------- | -------- | ------- |
| 0.87473 | 0.94 |  | 0.89 |  |  |
| Q        | SE       | SP      | PPV      | NPV      |         |
|  | 0.95  | 0.81 |  |   |         |

Mention: some problems lead to the interrupt during the training process, this is the result before interrupt.



###  6mA_D.melanogaster (ICjHp)

```json
{
    "batch_size": 128,
    "lr": 0.0001,
    "dropout": 0.3,
    "alpha": [
        0.4,
        0.6
    ]
}
```

| ACC      | AUC      | MCC     | F1       | F2       | F3      |
| -------- | -------- | ------- | -------- | -------- | ------- |
| 0.92109 | 0.969753 | 0.842273 | 0.920501 | 0.916392 | 0.91503 |
| Q        | SE       | SP      | PPV      | NPV      |         |
| 0.92109 | 0.913673  | 0.928508 | 0.927431 | 0.914935  |         |



###  6mA_Xoc BLS256 (owHB7)

```json
{
    "batch_size": 64,
    "lr": 0.0001,
    "dropout": 0.7,
    "alpha": [
        0.5,
        0.5
    ]
}
```

| ACC      | AUC      | MCC     | F1       | F2       | F3      |
| -------- | -------- | ------- | -------- | -------- | ------- |
| 0.882131 | 0.951329 | 0.764302 | 0.881518 | 0.878778 | 0.877868 |
| Q        | SE       | SP      | PPV      | NPV      |         |
| 0.882131 | 0.876961  | 0.887301 | 0.886123 | 0.87822  |         |