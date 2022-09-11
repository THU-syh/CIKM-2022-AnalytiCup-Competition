# For CIKM 2022 AnalytiCup Competition
Our code was explored and modified on the CIKM competition branch of the FS framework.
- To run the code:
```
    python federatedscope/main.py --cfg federatedscope/gfl/baseline/isolated_cikmcup_impratio.yaml --client_cfg federatedscope/gfl/baseline/cikmcup_per_client_normalize.yaml
```
- Brief introduction of the developed algorithm:

    We have tried a variety of federated learning algorithms, but unfortunately these methods have not shown good results. At present, our best result on the leaderboard is island training, but compared with the official baseline, certain configuration adjustments have been made. Such as learning rate of each client, Batchsize, LocalEpoch, etc. In addition, for some regression tasks, we counted the mean and standard deviation of y on the training set, and performed corresponding normalization and restoration operations during the training and prediction process.

