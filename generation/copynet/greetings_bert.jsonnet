local target_namespace = "target_tokens";
local transformer_model = "bert-base-cased";
local transformer_hidden_size = 768;
local epochs = 10;

{
    "dataset_reader": {
        "type": "copynet_seq2seq",
        "target_namespace": target_namespace,
        "source_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model,
        },
        "target_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model,
            "add_special_tokens": false,
        },
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model,
            },
        },
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
        // "encoder": {
        //     "type": "pass_through",
        //     "input_dim": transformer_hidden_size,
        // },
        "encoder": {
            "type": "lstm",
            "input_size": transformer_hidden_size,
            "hidden_size": 100,
            "num_layers": 2,
            "dropout": 0,
            "bidirectional": true
        },
        "attention": {
            "type": "bilinear",
            "vector_dim": 200,
            "matrix_dim": 200,
        },
        // "attention": {
        //     "type": "bilinear",
        //     "vector_dim": transformer_hidden_size,
        //     "matrix_dim": transformer_hidden_size,
        // },
        "target_embedding_dim": 10,
        "beam_size": 3,
        "max_decoding_steps": 20,
        "token_based_metric": {
            "type": "token_sequence_accuracy",
        },
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
            "lr": 0.02,
        },
        // "optimizer": {
        //     "type": "huggingface_adamw",
        //     "weight_decay": 0.0,
        //     // "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
        //     "lr": 2e-5,
        //     "eps": 1e-8
        // },
        // "learning_rate_scheduler": {
        //     "type": "slanted_triangular",
        //     "num_epochs": epochs,
        //     "cut_frac": 0.2,
        // },
        "learning_rate_scheduler": {
            "type": "cosine",
            "t_initial": 5,
            "eta_mul": 0.9
        },
        "grad_clipping": 1.0,
        "num_epochs": epochs,
        "cuda_device": 0,
    }
}
