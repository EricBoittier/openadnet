import pandas as pd

train         = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TRAIN.csv")
test          = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_BLINDED.csv")
train_counter  = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_counter-assay_TRAIN.csv")
test_structure = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_structure_TEST_BLINDED.csv")
train_single   = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_single_concentration_TRAIN.csv")