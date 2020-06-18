local target_namespace = "target_tokens";

{
    "dataset_reader": {
        "target_namespace": target_namespace,
        "type": "copynet_seq2seq",
        "source_token_indexers": {
            "tokens": {
                "type": "single_id",
                "namespace": "source_tokens"
            },
        }
    },
    "vocabulary": {
        "min_count": {
            "source_tokens": 4,
            "target_tokens": 4
        }
    },
    "train_data_path": "generation/copynet/data/greetings_train.tsv",
    "validation_data_path": "generation/copynet/data/greetings_validation.tsv",
    "model": {
        "type": "copynet_seq2seq",
        "source_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "vocab_namespace": "source_tokens",
                    "embedding_dim": 25,
                    "trainable": true
                },
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 25,
            "hidden_size": 50,
            "num_layers": 2,
            "dropout": 0,
            "bidirectional": true
        },
        "attention": {
            "type": "bilinear",
            "vector_dim": 100,
            "matrix_dim": 100
        },
        "target_embedding_dim": 10,
        "beam_size": 3,
        "max_decoding_steps": 20,
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size" : 32,
        }
    },
    "trainer": {
        "optimizer": {
            "type": "sgd",
            "lr": 0.1
        },
        "num_epochs": 2,
        "cuda_device": -1,
    }
}
