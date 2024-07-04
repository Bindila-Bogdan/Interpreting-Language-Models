import os
from datasets import load_dataset

class DatasetLoader:
    def __init__(self, directory="../data/", dataset_name="nyu-mll/blimp"):
        self.dataset_name = dataset_name
        self.directory = directory

    def __store_get_data(self):
        try:
            dataset = load_dataset(self.dataset_name, self.sub_dataset_name)
        except ValueError as e:
            print(e)
            return

        dataset.cache_files
        dataset["train"].to_json(self.storage_path)
        print(f"The dataset is stored at {self.directory}")

        return dataset

    def __load_from_local(self):
        data_files = {"train": self.storage_path}
        data = load_dataset("json", data_files=data_files)

        return data

    def load_data(self, sub_dataset_name):
        self.sub_dataset_name = sub_dataset_name
        self.storage_path = f"{self.directory}{self.dataset_name.split('/')[1]}_{self.sub_dataset_name}.json"

        if os.path.exists(self.storage_path):
            return self.__load_from_local()
        else:
            return self.__store_get_data()
