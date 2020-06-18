local target_namespace = "target_tokens";
local transformer_model = "bert-base-cased";
local hidden_size = 768;

{
    "dataset_reader": {
        "type": "copynet_seq2seq",
        "target_namespace": target_namespace,
        "source_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model,
        },
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model,
            },
        }
    },
    "vocabulary": {
        "min_count": {
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
                    "type": "pretrained_transformer",
                    "model_name": transformer_model,
                    "train_parameters": false,
                },
            }
        },
        "encoder": {
            "type": "pass_through",
            "input_dim": hidden_size,
        },
        "attention": {
            "type": "bilinear",
            "vector_dim": hidden_size,
            "matrix_dim": hidden_size,
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
