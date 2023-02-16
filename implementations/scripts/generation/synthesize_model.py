from scripts.tools.architecture_utils import ArchitectureUtils
from scripts.prediction.custom_objects import custom_objects
from scripts.logger.logger import Logger
from collections import defaultdict
import numpy as np
from scripts.mutation.comet_mutation_operators import copy_model_from_json
from scripts.generation.layer_pools import LAYERLIST, LayerInputNdim
import os
import traceback
import copy
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
NUM_SHAPE = 5
CONSIDER_SHAPE = True


class ModelDeduplication(object):
    def __init__(self):
        self.node_diversity = {}  # {layer_class1: [config1, config2], layer_class2: [config1, config2]}
        self.origin_node_diversity = {}
        self.edge_diversity = {}  # [(A, B), (B, A), (A, A), ...]
        self.input_diversity = {}  # this field is designed for future improvement
        self.node_config_count = {}  # {layer_class1: [0, 0, 0], layer_class2: [0, 0, 0]}
        self.layer_sequence = defaultdict(list)  # {layer_class1: [layer_class2, layer_class2, ...], layer_class2: [...], ...}
        self.input_properties = defaultdict(dict)  # {layer_class1: {"ndims": [], "dtype": [], ...}, ...}
        self.logger = Logger()
        from scripts.generation.layer_pools import LAYERLIST
        # get the layer type space
        self.total_layers = {}
        for layer_type in LAYERLIST:
            target_layers = LAYERLIST[layer_type]
            for layer_name, layer_func in target_layers.available_layers.items():
                self.total_layers[layer_name] = layer_func

    def _get_config(self, layer_class):
        if layer_class not in self.node_config_count:
            config_idx = np.random.randint(len(self.origin_node_diversity[layer_class]))
            return self.origin_node_diversity[layer_class][config_idx]
        possible_config_list = []
        for idx, config_count in enumerate(self.node_config_count[layer_class]):
            if config_count == 0:
                possible_config_list.append(self.node_diversity[layer_class][idx])
        if len(possible_config_list) != 0:
            return np.random.choice(possible_config_list)
        else:
            config_idx = np.random.randint(len(self.node_diversity[layer_class]))
            return self.node_diversity[layer_class][config_idx]

    @staticmethod
    def analyze_model(model):
        """
        Load, Analyze, Log The Layer API Call Diversity Inside The Given Model
        Arguments:
            :model: The keras model
        Return:
            Nothing...
        """
        node_diversity = ArchitectureUtils.extract_nodes(model)
        edge_diversity = ArchitectureUtils.extract_edges(model)
        input_diversity = ArchitectureUtils.extract_inputs(model)
        print(f"The Total Number Of Model's Layer Is: {len(model.layers)}")
        print(f"The Total Number Of Model's Weights Is: {model.count_params()}")
        print(f"The Total Number Of Layer Type Is: {len(node_diversity)}, The Total Number Of Edge Is: {len(edge_diversity)}")
        for layer_type in node_diversity:
            print(f"For Layer: {layer_type}, Total Number Of Unique Configuration Is: {len(node_diversity[layer_type])}")
        return node_diversity, edge_diversity, input_diversity

    @staticmethod
    def compare_edge(old, new):
        distance_edge = []
        for edge in new:
            if edge not in old:
                distance_edge.append(edge)
                print(f"Find New Edge: {edge}")
        return distance_edge, len(distance_edge)

    @staticmethod
    def compare_node(old, new):
        count = 0
        distance_node = defaultdict(list)
        for layer_class in new:
            if layer_class not in old:
                print(f"Find New Layer Type: {layer_class}")
                distance_node[layer_class] = copy.deepcopy(new[layer_class])
                count += len(distance_node[layer_class])
                continue
            for layer_config in new[layer_class]:
                if layer_config not in old[layer_class]:
                    distance_node[layer_class].append(layer_config)
                    print(f"Find New Layer Config: {layer_config}")
                    count += 1
        return distance_node, count

    @staticmethod
    def compare_input(old, new):
        """
        Compare the input diversity between old diversity and new diversity
        old, new: {"layer_class1": {"ndims": [], "dtype": [], "shape": []}, ...}
        Arguments:
            :old: old diversity state.
            :new: newly collected diversity.
        Return:
            distance_input: {"layer_class1": {"ndims": [], "dtype": [], "shape": []}, ...}
            new_input: boolean means whether new diversity has been found.
        """
        count = 0
        distance_input = {}
        for layer_class in new:
            new_ndims = set(new[layer_class]["ndims"])
            new_dtype = set(new[layer_class]["dtype"])
            new_shape = set(new[layer_class]["shape"])
            if layer_class not in old:
                print(f"Find New Layer: {layer_class}")
                distance_input[layer_class] = copy.deepcopy(new[layer_class])
                count += len(new_ndims) + len(new_dtype) + len(new_shape)
            else:
                old_ndims = set(old[layer_class]["ndims"])
                old_dtype = set(old[layer_class]["dtype"])
                old_shape = set(old[layer_class]["shape"])
                distance_input[layer_class] = {"ndims": [], "dtype": [], "shape": []}
                if len(new_ndims - old_ndims) > 0:
                    distance_input[layer_class]["ndims"] = list(new_ndims - old_ndims)
                    count += len(new_ndims - old_ndims)
                    print(f"Find New NDims: {new_ndims-old_ndims} For Layer: {layer_class}")
                if len(new_dtype - old_dtype) > 0:
                    distance_input[layer_class]["dtype"] = list(new_dtype - old_dtype)
                    count += len(new_dtype - old_dtype)
                    print(f"Find New DType: {new_dtype-old_dtype} For Layer: {layer_class}")
                if CONSIDER_SHAPE is True:
                    shape_distance = min(len(new_shape), 5) - min(len(old_shape), 5)
                    if shape_distance > 0:
                        count += shape_distance
                        remain_shape = list(new_shape - old_shape)
                        selected_shape = np.random.choice(remain_shape, shape_distance, replace=False)
                        distance_input[layer_class]["shape"] = selected_shape
                        print(f"Find New Shape: {selected_shape} For Layer: {layer_class}")
        return distance_input, count

    def compare_diversity(self, diversity1, diversity2):
        """
        Compare the diversity between original model and deduplicated model.
        Arguments:
            :diversity1: (node_diversity, edge_diversity, input_diversity)
            :diversity2: (node_diversity, edge_diversity, input_diversity)
        """
        old_node, old_edge, old_input = diversity1
        new_node, new_edge, new_input = diversity2
        self.logger.info(f"Checking If There Is Something Missed By Deduplicated Model")
        new_node, node_count = self.compare_node(new_node, old_node)
        new_edge, edge_count = self.compare_edge(new_edge, old_edge)
        new_input, input_count = self.compare_input(new_input, old_input)
        return new_node, new_edge, new_input, node_count+edge_count+input_count

    def get_uncovered_edge(self):
        uncovered_edge = []  # [(l1, l2), (l1, l3), ...]
        for layer_class in self.layer_sequence:
            for right_class in self.layer_sequence[layer_class]:
                uncovered_edge.append((layer_class, right_class))
        return uncovered_edge

    def get_uncovered_layer(self):
        uncovered_layer = []  # {layer_class1, layer_class2, ...}
        uncovered_merge_layer = []
        for layer_class in self.node_config_count:
            if layer_class in LAYERLIST["LMerg"].available_layers:
                if np.prod(self.node_config_count[layer_class]) == 0:
                    uncovered_merge_layer.append(layer_class)
                continue
            if np.prod(self.node_config_count[layer_class]) == 0:
                uncovered_layer.append(layer_class)
        return uncovered_layer, uncovered_merge_layer

    def get_uncovered_input(self):
        uncovered_ndims = {}
        uncovered_dtype = {}
        uncovered_shape = {}
        for layer_class, item in self.input_properties.items():
            if "ndims" in item:
                uncovered_ndims[layer_class] = item["ndims"]
            if "dtype" in item:
                uncovered_shape[layer_class] = item["dtype"]
            if "shape" in item:
                uncovered_shape[layer_class] = item["shape"]
        return uncovered_ndims, uncovered_dtype, uncovered_shape

    def append_merge_layer_to_model(self, model, layer_class):
        target_json = model.to_json()
        del model
        model_copy = copy_model_from_json(target_json)
        new_layer = self.total_layers[layer_class]
        config = self._get_config(layer_class)
        from scripts.generation.layer_pools import ConfigurationUtils
        inserted_layer = ConfigurationUtils.from_config(new_layer, config)
        layer1_name, layer2_name = ArchitectureUtils.find_two_layer_with_same_output_shape(model_copy)
        if layer1_name is None or layer2_name is None:
            raise ValueError("Failed to add the merging layer, no suitable tensor to find")
        # Step 3: merge the output of two layers.
        model_copy = ArchitectureUtils.merge_two_layers(model_copy, inserted_layer, layer1_name, layer2_name)
        print(f"Successfully Adding The Merging Layer: {layer_class}")
        return model_copy

    def append_layer_to_model(self, model, layer_class, required_input_shape=None):
        from scripts.mutation.comet_mutation_operators import NL_mut
        target_json = model.to_json()
        del model
        model_copy = copy_model_from_json(target_json)
        new_layer = self.total_layers[layer_class]
        selection_pair = {model_copy.layers[-1].name: new_layer}
        config = self._get_config(layer_class)
        config_pair = {model_copy.layers[-1].name: config}
        # Note that the parameter: selected_layer_name_list is of no use
        model_copy = NL_mut(model_copy, selection_pair, selected_layer_name_list=[], config_pair=config_pair, required_input_shape=required_input_shape, revert_shape=False)
        return model_copy

    @staticmethod
    def change_layer_shape(model, layer_idx):
        from scripts.mutation.comet_mutation_operators import InputMutationUtils
        target_json = model.to_json()
        del model
        model_copy = copy_model_from_json(target_json)
        layer_name = model_copy.layers[layer_idx].name
        input_shape = model_copy.layers[layer_idx].input.shape
        generated_shape = ArchitectureUtils.generate_shape_by_ndims(len(input_shape), min=5, max=20)
        selected_layer_shape_pair = {layer_name: (generated_shape, [])}
        model_copy = InputMutationUtils.mshape(model_copy, selected_layer_shape_pair, revert_shape=False)
        return model_copy

    @staticmethod
    def change_layer_dims(model, layer_idx, ndim):
        from scripts.mutation.comet_mutation_operators import InputMutationUtils
        target_json = model.to_json()
        del model
        model_copy = copy_model_from_json(target_json)
        layer_name = model_copy.layers[layer_idx].name
        selected_layer_ndims_pair = {layer_name: (ndim, [])}
        model_copy = InputMutationUtils.mdims(model_copy, selected_layer_ndims_pair=selected_layer_ndims_pair, revert_shape=False)
        return model_copy

    def insert_edge_to_model(self, model, edge):
        target_json = model.to_json()
        del model
        model_copy = copy_model_from_json(target_json)
        from scripts.tools.architecture_utils import ArchitectureUtils
        layer1_class, layer2_class = edge
        layer1_name = ArchitectureUtils.pick_layer_by_class(model_copy, layer1_class)
        layer2_name = ArchitectureUtils.pick_layer_by_class(model_copy, layer2_class)
        # Note that the edge should be: layer1_name -> layer2_name
        self.logger.info(f"Trying Adding Edge: {edge} by Connecting {layer1_name} and {layer2_name}")
        model_copy = ArchitectureUtils.connect_two_layers(model_copy, layer1_name, layer2_name)
        self.logger.info(f"Successfully Add Edge: {edge} by Connecting {layer1_name} and {layer2_name}")
        return model_copy

    def initiate_state(self, node_diversity, edge_diversity, input_diversity):
        self.layer_sequence = defaultdict(list)
        self.node_config_count = defaultdict(list)
        self.input_properties = defaultdict(dict)
        self.node_diversity = node_diversity
        self.edge_diversity = edge_diversity
        self.input_diversity = input_diversity
        for edge in edge_diversity:
            l1, l2 = edge
            self.layer_sequence[l1].append(l2)
        for layer_class, config_list in node_diversity.items():
            self.node_config_count[layer_class] = []
            for config in config_list:
                self.node_config_count[layer_class].append(0)
        for layer_class in input_diversity:
            for property_name, item in input_diversity[layer_class].items():
                if len(item) > 0:
                    self.input_properties[layer_class] = {}
                    self.input_properties[layer_class][property_name] = item
        print(self.layer_sequence)
        print(self.node_config_count)
        print(self.input_properties)

    def reformat_model(self, model):
        previous_layer_class = model.layers[-1]
        if len(self.layer_sequence[previous_layer_class]) > 0:
            new_layer_class = np.random.choice(self.layer_sequence[previous_layer_class])
            model = self.append_layer_to_model(model, new_layer_class)
            return model
        uncovered_layer, uncovered_merge_layer = self.get_uncovered_layer()
        if len(uncovered_layer) > 0:
            layer_class = np.random.choice(uncovered_layer)
            model = self.append_layer_to_model(model, layer_class)
            return model
        uncovered_input_ndims, uncovered_input_dtype, uncovered_input_shape = self.get_uncovered_input()
        print(uncovered_input_ndims, uncovered_input_dtype, uncovered_input_shape)
        if len(uncovered_input_ndims) > 0:
            layer_class = np.random.choice(list(uncovered_input_ndims.keys()))
            target_ndim = np.random.choice(uncovered_input_ndims[layer_class])
            model = self.append_layer_to_model(model, layer_class)
            layer_idx = -1
            model = self.change_layer_dims(model, layer_idx, target_ndim)
            return model
        if len(uncovered_input_dtype) > 0:
            pass
        if CONSIDER_SHAPE is True:
            if len(uncovered_input_shape) > 0:
                layer_class = np.random.choice(list(uncovered_input_shape.keys()))
                target_shape = np.random.choice(list(uncovered_input_shape[layer_class]))
                ndim = target_shape.count(",")+1
                generated_shape = ArchitectureUtils.generate_shape_by_ndims(ndim, min=5, max=20)
                model = self.append_layer_to_model(model, layer_class, required_input_shape=generated_shape)
                return model
        uncovered_edge = self.get_uncovered_edge()
        if len(uncovered_edge) > 0:
            # print(f"The Edge Is Not Fully Repaired!")
            edge_idx = np.random.randint(len(uncovered_edge))
            edge = uncovered_edge[edge_idx]
            model = self.insert_edge_to_model(model, edge)
            return model
        if len(uncovered_merge_layer) > 0:
            layer_class = np.random.choice(uncovered_merge_layer)
            model = self.append_merge_layer_to_model(model, layer_class)
            return model
        return model

    def run(self, model_path):
        import keras
        self.logger.info("Loading The Original Model!")
        origin_model = keras.models.load_model(model_path, custom_objects=custom_objects())
        origin_model.summary()
        target_diversity = self.analyze_model(origin_model)
        import keras
        x = keras.layers.Input(origin_model.input_shape[1:])
        if origin_model.__class__.__name__ == "Sequential":
            # The first layer is not InputLayer
            first_layer = origin_model.layers[0]
        else:
            first_layer = origin_model.layers[1]
        y = first_layer(x)
        new_model = keras.models.Model(x, y)
        achieved_diversity = self.analyze_model(new_model)
        distance_node, distance_edge, distance_input, distance_value = self.compare_diversity(target_diversity, achieved_diversity)
        self.origin_node_diversity = target_diversity[0]
        # Set The Goal
        try_time = 0
        while distance_value > 0:
            self.initiate_state(distance_node, distance_edge, distance_input)
            self.logger.info("Start Deduplication!")
            temp_model = self.reformat_model(new_model)
            temp_diversity = self.analyze_model(temp_model)
            temp_distance_node, temp_distance_edge, temp_distance_input, temp_distance_value = self.compare_diversity(target_diversity, temp_diversity)
            print(distance_value, temp_distance_value)
            # check if the distance is smaller
            if temp_distance_value < distance_value:
                print("Updating Distance")
                distance_node, distance_edge, distance_input, distance_value = \
                    temp_distance_node, temp_distance_edge, temp_distance_input, temp_distance_value
                achieved_diversity = temp_diversity
                new_model = temp_model
            try_time += 1
            if try_time > 100:
                raise ValueError("Try Time Exceed!")
        return new_model, target_diversity, achieved_diversity


if __name__ == "__main__":
    import datetime
    from scripts.tools import utils
    start_time = datetime.datetime.now()

    origin_model_path = "/root/data/origin_models"
    new_model_path = "/root/data/working_dir/synthesized_models"
    if not os.path.exists(new_model_path):
        os.makedirs(new_model_path)
    origin_analyze_result = {}
    new_analyze_result = {}
    for path in os.listdir(origin_model_path):
        if not path.endswith(".h5"):
            continue
        model_name = path.split(".h5")[0]
        import keras
        origin_model = keras.models.load_model(os.path.join(origin_model_path, path))
        origin_analyze_result[model_name] = {}
        new_analyze_result[model_name] = {}
        print(f"=============== Working On Model: {path} ===================")
        s_time = time.time()
        new_model, origin_diversity, new_diversity = None, None, None
        e_time = time.time()
        while e_time - s_time <= 60*5 and new_model is None:
            try:
                modelDeduplication = ModelDeduplication()
                new_model, origin_diversity, new_diversity = modelDeduplication.run(os.path.join(origin_model_path, path))
            except:
                print(traceback.format_exc())
                new_model = None
            e_time = time.time()
        if new_model is None:
            print(f"Failed When Synthesizing Model: {model_name}")
        origin_analyze_result[model_name]["num_layer"] = len(origin_model.layers)
        new_analyze_result[model_name]["num_layer"] = len(new_model.layers)
        origin_analyze_result[model_name]["diversity"] = origin_diversity
        new_analyze_result[model_name]["diversity"] = new_diversity

        def count_config(node_diversity):
            config_count = 0
            for layer_class, config_list in node_diversity.items():
                config_count += len(config_list)
            return config_count

        origin_analyze_result[model_name]["layer_type"] = len(origin_diversity[0])
        origin_analyze_result[model_name]["layer_sequence"] = len(origin_diversity[1])
        origin_analyze_result[model_name]["layer_parameter"] = count_config(origin_diversity[0])

        new_analyze_result[model_name]["layer_type"] = len(new_diversity[0])
        new_analyze_result[model_name]["layer_sequence"] = len(new_diversity[1])
        new_analyze_result[model_name]["layer_parameter"] = count_config(new_diversity[0])

        model_json = new_model.to_json()
        with open(os.path.join(new_model_path, f"{path[:-5]}-simplified.json"), "w") as file:
            file.write(model_json)
        new_model.save(os.path.join(new_model_path, f"{path[:-5]}-simplified.h5"))

    print("============ Summary ================")
    for model_name in origin_analyze_result:
        origin_result = origin_analyze_result[model_name]
        new_result = new_analyze_result[model_name]
        print(f"For Model: {model_name}:")

        def print_result(result):
            print(
                f"{result['num_layer']} Layers \n"
                f"{result['layer_type']} Layer Types \n"
                f"{result['layer_sequence']} Layer Sequences \n"
                f"{result['layer_parameter']} Layer Parameters \n"
                )
        print("Origin Model")
        print_result(origin_result)
        print("Deduplicated Model")
        print_result(new_result)
        modelDeduplication = ModelDeduplication()
        distance_node, distance_edge, distance_input, distance_value = modelDeduplication.compare_diversity(origin_result["diversity"], new_result["diversity"])
        print(distance_node, distance_edge, distance_value)
    end_time = datetime.datetime.now()
    time_delta = end_time - start_time
    h, m, s = utils.ToolUtils.get_HH_mm_ss(time_delta)
    print(f"Model Deduplication Is Finished: Time used: {h} hour,{m} min,{s} sec")
