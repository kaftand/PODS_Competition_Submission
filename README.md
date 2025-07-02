# PODS Competition Submission

To validate my results, do the following steps
1. create conda env & activate
```
conda env create -f environment.yml -n kaftand_PODS
conda activate kaftand_PODS
```
2. run local_test
```
python local_test.py
```

In my runs, I generally get ~0.054 rmse for the CTR and ~0.29 for CONV, slightly worse than the competition dataset. Your mileage may vary due to the fact that I don't use a random seed to initialize rates.

## Note on why my development score was way worse than my final score
I believe that the development datasets were shuffled (see my forum post here: https://www.codabench.org/forums/7466/1185/ ). I trained my model on unshuffled data using the provided datasets in the competition overview page. 
My model heavily relies on long term (over 1000s of samples) trends in the very noisy data. So if it learns on unshuffled data, and predicts on shuffled data, it will perform poorly. I submitted the exact same file in the development and final phases, indicating to me that the data during the development phase was shuffled, and the data during the final phase was not.

I have more or less validated this by playing with the local test file, and I encourage others to do the same.
