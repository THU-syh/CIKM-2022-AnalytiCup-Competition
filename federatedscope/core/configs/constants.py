"""Configuration file for composition of different aggregators, messages,
handlers, etc.

    - The method `local` indicates that the clients only locally
    train their model without sharing any training related information
    - The method `global` indicates that the only one client locally trains
    using all data

"""

AGGREGATOR_TYPE = {
    "local": "no_communication",  # the clients locally train their model
    # without sharing any training related info
    "global": "no_communication",  # only one client locally train all data,
    # i.e., totally global training
    "fedavg": "clients_avg",  # FedAvg
    "pfedme": "server_clients_interpolation",  # pFedMe,  + server-clients
    # interpolation
    "ditto": "clients_avg",  # Ditto
    "ditto_finetune": "clients_avg_finetune",
    "fedsageplus": "clients_avg",
    "gcflplus": "clients_avg",
    "fedopt": "fedopt",
    "adaptupdate": "clients_avg",
}

CLIENTS_TYPE = {
    "local": "normal",
    "fedavg": "normal",  # FedAvg
    "pfedme": "normal_loss_regular",  # pFedMe, + regularization-based local
    # loss
    "ditto": "normal",  # Ditto, + local training for distinct personalized
    "ditto_finetune": "finetune",
    # models
    "fedsageplus": "fedsageplus",  # FedSage+ for graph data
    "gcflplus": "gcflplus",  # GCFL+ for graph data
    "gradascent": "gradascent",
    "adaptupdate": "normal",
}

SERVER_TYPE = {
    "local": "normal",
    "fedavg": "normal",  # FedAvg
    "pfedme": "normal",  # pFedMe, + regularization-based local loss
    "ditto": "normal",  # Ditto, + local training for distinct personalized
    "ditto_finetune": "finetune",
    # models
    "fedsageplus": "fedsageplus",  # FedSage+ for graph data
    "gcflplus": "gcflplus",  # GCFL+ for graph data
    "adaptupdate": "normal",
}
