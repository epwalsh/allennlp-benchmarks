local transformer_model = 'bert-base-cased';

local epochs = 3;
local batch_size = 8;

{
  "dataset_reader": {
      "type": "sharded",
      "base_reader": {
          "type": "transformer_squad",
          "transformer_model_name": transformer_model,
          "skip_invalid_examples": true,
      },
  },
  "validation_dataset_reader": {
      "type": "sharded",
      "base_reader": {
          "type": "transformer_squad",
          "transformer_model_name": transformer_model,
          "skip_invalid_examples": false,
      },
  },
  "train_data_path": "https://github.com/epwalsh/allennlp-benchmarks/raw/master/data/squad_v1_1/train.tar.gz",
  "validation_data_path": "https://github.com/epwalsh/allennlp-benchmarks/raw/master/data/squad_v1_1/validation.tar.gz",
  "vocabulary": {"type": "empty"},
  "model": {
      "type": "transformer_qa",
      "transformer_model_name": transformer_model,
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": batch_size
    }
  },
  "trainer": {
    "opt_level": "O2",
    "optimizer": {
      "type": "huggingface_adamw",
      "weight_decay": 0.0,
      "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
      "lr": 2e-5,
      "eps": 1e-8
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": epochs,
      "cut_frac": 0.1,
    },
    "grad_clipping": 1.0,
    "num_epochs": epochs,
  },
  "random_seed": 42,
  "numpy_seed": 42,
  "pytorch_seed": 42,
  "distributed": {"cuda_devices": [0, 1]},
}
