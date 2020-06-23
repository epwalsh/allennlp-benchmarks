import os
import json

from allennlp.common.file_utils import cached_path, open_compressed


NUM_SHARDS = 8
TRAIN_DATA = "https://allennlp.s3.amazonaws.com/datasets/squad/squad-train-v1.1.json"
VALIDATION_DATA = "https://allennlp.s3.amazonaws.com/datasets/squad/squad-dev-v1.1.json"
OUTPUT_BASE_DIR = "./data/squad_v1_1"


def shard_dataset(dataset_path: str, output_directory: str):
    print(f"Reading data from {dataset_path}")
    with open_compressed(cached_path(dataset_path)) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json["data"]

    shards = [{"data": []} for _ in range(NUM_SHARDS)]

    i = 0
    for article in dataset:
        for paragraph_json in article["paragraphs"]:
            context = paragraph_json["context"]
            for question_answer in paragraph_json["qas"]:
                shard_num = i % NUM_SHARDS
                shard = shards[shard_num]
                shard["data"].append(
                    {"paragraphs": [{"context": context, "qas": [question_answer]}]}
                )
                i += 1

    for i, shard in enumerate(shards):
        print(f"Shard {i}: {len(shard['data'])} question answers")

    for i, shard in enumerate(shards):
        shard_file_path = os.path.join(output_directory, f"shard_{i}.json")
        print(f"Writing shard {i} to {shard_file_path}")
        with open(shard_file_path, "w") as shard_file:
            json.dump(shard, shard_file)


if __name__ == "__main__":
    shard_dataset(TRAIN_DATA, OUTPUT_BASE_DIR + "/train")
    shard_dataset(VALIDATION_DATA, OUTPUT_BASE_DIR + "/validation")
