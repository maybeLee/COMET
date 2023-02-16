import argparse
import configparser
import sys
from scripts.mutation.structure_mutation_generators import generate_model_by_model_mutation
import os
import pickle
import dill
from scripts.tools.architecture_utils import ArchitectureUtils
import redis


class ModelGenerator(object):
    """
    This class is used to 1) mutate a model and 2) save a model
    """

    def __init__(self, flags):
        self.config = configparser.ConfigParser()
        self.config_name = flags.config_name
        self.config_path = f"./config/{self.config_name}"
        self.config.read(self.config_path)
        self.architecture_measure_path = flags.architecture_measure_path
        self.architecture_measure = None
        self.new_model_name = flags.new_model_name
        pool = redis.ConnectionPool(host=self.config['redis']['host'], port=self.config['redis']['port'],
                                    db=self.config['redis'].getint('redis_db'))
        self.redis_conn = redis.Redis(connection_pool=pool)
        self.config_gpus()
        self.load_architecture_measure()

    def config_gpus(self):
        parameters = self.config['parameters']
        gpu_ids = parameters["gpu_ids"]
        gpu_list = gpu_ids.split(",")
        if len(gpu_list) >= 2:
            gpu_list = gpu_list[1:]
        gpu_ids = ",".join(gpu_list)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    def load_architecture_measure(self):
        with open(self.architecture_measure_path, "rb") as file:
            self.architecture_measure = dill.load(file)

    def save_architecture_measure(self):
        with open(self.architecture_measure_path, "wb") as file:
            dill.dump(self.architecture_measure, file)

    def load_origin_model(self, model_path):
        origin_model = ArchitectureUtils.load_json(model_path)
        return origin_model

    def mutate_model(self, origin_model):
        import keras
        mutated_model = generate_model_by_model_mutation(
            model=origin_model,
            operator=flags.mutate_op,
            mutation_operator_mode=flags.mutation_operator_mode,
            architecture_measure=self.architecture_measure
        )
        if mutated_model is None:
            raise Exception("Error: Model mutation using {} failed".format(flags.mutate_op))
        if int(keras.__version__.split(".")[1]) >= 7:
            mutated_model._name = self.new_model_name
        else:
            mutated_model.name = self.new_model_name
        return mutated_model

    def save_model(self, model, model_path):
        """
        If a DL model has multiple outputs, we merge them together, then convert it to a (300,) tensor
        """
        import keras
        # Step 2.1: Model input processing
        # For pytorch, it cannot automatically infer the input shape, therefore, we need to send to it through redis
        shape = list(model.input_shape)
        shape[0] = 1
        self.redis_conn.hset(f"prediction_{self.new_model_name}", "shape", pickle.dumps(shape))
        x = model.inputs
        # Step 2.2: Model output processing
        # If model has multiple outputs, we merge them together by keras.layers.Add()
        actual_output_lists = []
        for model_output in model.outputs:
            layer_input = model_output
            input_shape = layer_input.shape.as_list()
            from scripts.generation.layer_tool_kits import LayerMatching
            for layer in LayerMatching.flatten_dense_output(input_shape):
                y = layer(layer_input)
                layer_input = y
            actual_output_lists.append(layer_input)
        if len(actual_output_lists) > 1:
            actual_output = keras.layers.Add()(actual_output_lists)
        else:
            actual_output = actual_output_lists[0]
        # Step 2.3: Create and save the model.
        actual_model = keras.Model(x, actual_output)
        if int(keras.__version__.split(".")[1]) >= 7:
            actual_model._name = self.new_model_name
        else:
            actual_model.name = self.new_model_name
        # actual_model.summary()
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        actual_model.save(model_path)


if __name__ == "__main__":

    """Parser of command args"""
    parse = argparse.ArgumentParser()
    parse.add_argument("--origin_model_path", type=str, help="model path")
    parse.add_argument("--new_model_path", type=str, help="the path of mutated model's path (h5)")
    parse.add_argument("--new_model_config_path", type=str, help="the path of mutated model's path (json)")
    parse.add_argument("--new_model_name", type=str, help="the name of generated model")
    parse.add_argument("--mutate_op", type=str, help="model mutation operator")
    parse.add_argument("--mutation_operator_mode", type=str, help="model mutation's mode")
    parse.add_argument("--architecture_measure_path", type=str, help="the path of architecture_measure class")
    parse.add_argument("--config_name", type=str)
    flags, unparsed = parse.parse_known_args(sys.argv[1:])

    modelGenerator = ModelGenerator(flags)
    origin_model = modelGenerator.load_origin_model(flags.origin_model_path)
    new_model = modelGenerator.mutate_model(origin_model)
    ArchitectureUtils.save_json(new_model, json_path=flags.new_model_config_path)
    modelGenerator.save_model(new_model, flags.new_model_path)
