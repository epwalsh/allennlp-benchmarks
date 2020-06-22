from allennlp_models.generation.predictors import Seq2SeqPredictor

predictor = Seq2SeqPredictor.from_path("/tmp/run1/model.tar.gz")
predictor.predict("It's Jaquane")["predicted_tokens"][0]

with open("./generation/copynet/data/greetings_validation.tsv") as validation_file:
    for line in validation_file:
        line = line.strip()
        source, target = line.split("\t")
        predicted_tokens = predictor.predict(source)["predicted_tokens"][0]
        target_tokens = [
            t.text for t in predictor._dataset_reader._target_tokenizer.tokenize(target)
        ]
        if predicted_tokens != target_tokens:
            print(predicted_tokens)
            print(target_tokens)
            break
        else:
            print(f"- {target}")
