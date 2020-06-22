from allennlp_models.rc.dataset_readers import SquadReader
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.params import Params

params = Params.from_file("./bidaf_1x_gpu.jsonnet")
reader_params = params.pop("dataset_reader")
reader: SquadReader = DatasetReader.from_params(reader_params)

train_data_path = params.pop("train_data_path")
instances = list(reader.read(train_data_path))
