# Seen Course prediction

## requirements
- implicit
- scipy
- numpy
- pandas


## how to use
First, all the csv files should be placed in a folder name `data`.
All the python should be executed within the folder where `data` is.
e.g
```
.
├── ALS.py
├── README.md
├── average_precision.py
├── bayesian.py
├── data
└── example.py
```

### Use ALS

```shell
python ALS.py
```

### Use bayesian estimation
```shell
python bayesian.py
```

the generate predicted csv `predict.csv` will also be in the `data` folder
