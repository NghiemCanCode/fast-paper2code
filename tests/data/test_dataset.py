import unittest
from src.data.datasets import GossipCopDataset
from pathlib import Path

class GossipCopDatasetTestCase(unittest.TestCase):
    def test_init_method(self):
        filepath = Path("../../data/prompt_data_response/ZsCoTGossipCopFakeEvi_2025-05-30_11-33-00/")

        train_dataset = GossipCopDataset(filepath, "train", "google-bert/bert-base-uncased", 128)
        print(train_dataset[0])

if __name__ == '__main__':
    unittest.main()
