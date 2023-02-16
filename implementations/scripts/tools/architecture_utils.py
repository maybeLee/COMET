from asyncio.log import logger
from itertools import combinations, combinations_with_replacement
import numpy as np
from scripts.logger.logger import Logger
from collections import defaultdict

mylogger = Logger()

import copy


class ArchitectureUtils:

    @staticmethod
    def save_json(model, json_path):
        model_json = model.to_json()
        with open(json_path, "w") as file:
            file.write(model_json)

    @staticmethod
    def load_json(save_path):
        from scripts.prediction.custom_objects import custom_objects
        import keras
        with open(save_path, "rb") as file:
            model_json = file.read()
        model = keras.models.model_from_json(model_json, custom_objects=custom_objects())
        return model

    @staticmethod
    def get_expand_drop_dim_layer(input_shape, target_shape):
        layer_list = []
        from scripts.generation.custom_layers import CustomExpandLayer, CustomDropDimLayer
        dim = len(input_shape)  # the dimension of tensor
        dim_distance = len(target_shape) - dim
        if dim_distance != 0:
            # If dimension inconsistency occur, we expand or drop dimension for input tensor
            if dim_distance > 0:
                # if dim_distance > 0, we expand the input
                while dim_distance > 0:
                    custom_expand_layer = CustomExpandLayer(axis=1)
                    layer_list.append(custom_expand_layer)
                    dim_distance -= 1
            elif dim_distance < 0:
                while dim_distance < 0:
                    custom_drop_dim_layer = CustomDropDimLayer(axis=1)
                    layer_list.append(custom_drop_dim_layer)
                    dim_distance += 1
            else:
                pass
        else:
            pass
        return layer_list

    @staticmethod
    def _expand_drop_dim(tensor, target_shape):
        input_shape = tensor.shape.as_list()
        layer_list = ArchitectureUtils.get_expand_drop_dim_layer(input_shape, target_shape)
        for _layer in layer_list:
            tensor = _layer(tensor)
        # check dim equal
        assert len(tensor.shape) == len(target_shape)
        return tensor

    @staticmethod
    def _pad_crop_shape(tensor, target_shape, mode="custom"):
        """
        Arguments:
            mode: we have two option: `custom` and `keras`,
                two option will have two different implementation while having the same specification.

        Return:
            tensor that has the same shape as target_shape
        """
        # we have two mode: custom, keras
        dim = len(tensor.shape.as_list())  # the dimension of tensor

        def check_shape(distance, padding_n, cropping_n):
            if distance > 0:
                # need to pad
                padding_n = [
                    distance // 2, distance - distance // 2
                ]
            elif distance < 0:
                distance = -distance
                cropping_n = [
                    distance // 2, distance - distance // 2
                ]
            else:
                pass
            return padding_n, cropping_n

        if mode == "custom":
            # We can also use customized layer to do so
            from scripts.generation.custom_layers import CustomPadLayer, CustomCropLayer
            padding, cropping = [], []
            for d in range(dim - 1):
                padding.append([0, 0])
                cropping.append([0, 0])
            i = 0
            for ts, ns in zip(target_shape[1:], tensor.shape.as_list()[1:]):
                distance = ts - ns
                padding[i], cropping[i] = check_shape(distance, padding[i], cropping[i])
                i += 1
            if np.any(padding):
                tensor = CustomPadLayer(padding=padding, constant_values=2)(tensor)
            if np.any(cropping):
                tensor = CustomCropLayer(cropping=cropping)(tensor)
        elif mode == "keras":
            # We directly use keras's function to do so
            import keras
            if dim == 3:
                # we are working on 3D tensor
                distance_0 = target_shape[1] - tensor.shape.as_list()[1]
                padding = [0, 0]
                cropping = [0, 0]
                padding, cropping = check_shape(distance_0, padding, cropping)
                if np.any(padding):
                    tensor = keras.layers.ZeroPadding1D(padding=padding)(tensor)
                if np.any(cropping):
                    tensor = keras.layers.Cropping1D(cropping=cropping)(tensor)
            elif dim == 4:
                # we are working on 4D tensor
                distance_0 = target_shape[1] - tensor.shape.as_list()[1]
                distance_1 = target_shape[2] - tensor.shape.as_list()[2]
                padding = [[0, 0], [0, 0]]
                cropping = [[0, 0], [0, 0]]
                padding[0], cropping[0] = check_shape(distance_0, padding[0], cropping[0])
                padding[1], cropping[1] = check_shape(distance_1, padding[1], cropping[1])
                if np.any(padding):
                    tensor = keras.layers.ZeroPadding2D(padding=padding)(tensor)
                if np.any(cropping):
                    tensor = keras.layers.Cropping2D(cropping=cropping)(tensor)
            elif dim == 5:
                # we are working on 5D tensor
                distance_0 = target_shape[1] - tensor.shape.as_list()[1]
                distance_1 = target_shape[2] - tensor.shape.as_list()[2]
                distance_2 = target_shape[3] - tensor.shape.as_list()[3]
                padding = [[0, 0], [0, 0], [0, 0]]
                cropping = [[0, 0], [0, 0], [0, 0]]
                padding[0], cropping[0] = check_shape(distance_0, padding[0], cropping[0])
                padding[1], cropping[1] = check_shape(distance_1, padding[1], cropping[1])
                padding[2], cropping[2] = check_shape(distance_2, padding[2], cropping[2])
                if np.any(padding):
                    tensor = keras.layers.ZeroPadding3D(padding=padding)(tensor)
                if np.any(cropping):
                    tensor = keras.layers.Cropping3D(cropping=cropping)(tensor)
        assert list(tensor.shape[1:]) == list(target_shape[1:])
        return tensor

    @staticmethod
    def reshape_tensor(tensor, target_shape, mode="custom"):
        """
        This is a magic method that can change tensor's shape to a target shape
        using keras.CroppingXD and CustomPadLayer/ZeroPaddingXD.
        Currently we only support operation for 3D (Batch, S, C), 4D, and 5D tensor.
        Arguments:
            tensor: the tensor to be cropped/pad.
            target_shape: the target shape we want to convert the tensor to.
        Return:
            the converted tensor
        Example:
            x = np.random.rand(10, 3, 3, 3)
            target_shape = (10, 10, 10, 3)
            new_x = ArchitectureUtils.reshape_tensor(x, target_shape)
            print(new_x.shape)
        """
        new_tensor = tensor
        if None in tensor.shape[1:] or None in target_shape[1:]:
            logger.info("We find some unknown shape, can only do dimension matching")
            # Step 1: expand/drop if the dimension is inconsistent
            new_tensor = ArchitectureUtils._expand_drop_dim(new_tensor, target_shape)
        else:
            # Step 1: expand/drop if the dimension is inconsistent
            new_tensor = ArchitectureUtils._expand_drop_dim(new_tensor, target_shape)
            # Step 2: pad/crop if the shape is inconsistent
            new_tensor = ArchitectureUtils._pad_crop_shape(new_tensor, target_shape, mode=mode)
        return new_tensor

    @staticmethod
    def trim_edge(edges, assert_dim=False, layer_class_dict=None):
        # trim edge with binary operator or inconsistent dimension
        merge_layer_class = ["Concatenate", "Average", "Maximum", "Minimum", "Add", "Subtract", "Multiply", "Dot",
                             "Dense"]
        trimmed_edge = []
        # Check if the layer belongs to the target class.
        from scripts.generation.layer_pools import LAYERLIST
        total_layers = []
        for layer_type in LAYERLIST:
            target_layers = LAYERLIST[layer_type]
            total_layers += list(target_layers.available_layers.keys())
        for edge in edges:
            edge_list = list(edge)
            layer1_class, layer2_class = edge_list[0], edge_list[1]
            if layer1_class in merge_layer_class:
                continue
            if layer2_class in merge_layer_class:
                continue
            if layer1_class not in total_layers or layer2_class not in total_layers:
                continue
            if assert_dim is True:
                if layer_class_dict is None:
                    raise Exception(
                        "The layer_class_dict is None when assert_dim is True. Please define the layer_class_dict")
                from scripts.generation.layer_pools import LayerInputNdim
                layer1_class_output_dim = set(layer_class_dict[layer1_class].values())
                connectable = len(layer1_class_output_dim.intersection(
                    set(LayerInputNdim[layer2_class]))) != 0  # True: connectable, False: inconnectable
                if connectable is False:
                    continue
            trimmed_edge.append(edge)
        return trimmed_edge

    @staticmethod
    def find_new_edges(model, covered_edges, max_num=10, mode="diverse"):
        """

        Arguments:
            :model: model to be analyzed
            :covered_edges: a list storing all covered edges [(s, e), ...].
            :max_num: insert [1, max_num) layer sequences
            :mode:
                diverse -> first iterate global layers, then iterate local layers.
                random -> randomly choose layers.
        Returns:
            existing edges, candidate edges
        """
        if model.__class__.__name__ == 'Sequential':
            layer_list = model.layers
        else:
            layer_list = model.layers[1:]  # ignore the first input layer

        # collect existing edges and all layer classes
        layer_class_dict = defaultdict(dict)
        # {layer1_class: {layer1_name: 2, layer2_name: 3}, layer2_class: {layer1_name: 3, layer2_name:3}}
        for layer in layer_list:
            layer_class_dict[layer.__class__.__name__][layer.name] = len(layer.output_shape)

        global_candidate_edges = []
        other_candidate_edges = []
        for edge in combinations_with_replacement(set(layer_class_dict.keys()), 2):
            """
            a = [1,2,3]
            combinations_with_replacement(a,2): 
            (1, 1)
            (1, 2)
            (1, 3)
            (2, 2)
            (2, 3)
            (3, 3)
            """
            s, e = edge
            if edge not in covered_edges:
                global_candidate_edges.append(edge)
            else:
                other_candidate_edges.append(edge)
            if s != e:
                # corner case when the start and end of the edge is not the same.
                edge1 = (e, s)
                if edge1 not in covered_edges:
                    global_candidate_edges.append(edge1)
                else:
                    other_candidate_edges.append(edge1)
        filtered_global_candidate_edges = ArchitectureUtils.trim_edge(global_candidate_edges, assert_dim=True,
                                                                      layer_class_dict=layer_class_dict)  # type: list
        filtered_other_candidate_edges = ArchitectureUtils.trim_edge(other_candidate_edges)  # type: list
        filtered_all_candidate_edges = filtered_global_candidate_edges + filtered_other_candidate_edges
        num_total_edges = len(filtered_all_candidate_edges)
        num_edge = np.random.randint(1, min(max_num + 1, (num_total_edges + 1) // 2 + 1))
        print(f"Choosing {num_edge} To Insert")
        selected_edge_list = []
        if len(filtered_global_candidate_edges) != 0 and mode == "diverse":
            # The reason that does not directly use random choice the edge list because np.random.choice does not support randomly choose
            if len(filtered_global_candidate_edges) >= num_edge:
                mylogger.info(f"Insert {num_edge} Global New Edges")
                random_idx = np.random.choice(len(filtered_global_candidate_edges), num_edge, replace=False)
                for idx in random_idx:
                    selected_edge_list.append(filtered_global_candidate_edges[idx])
                return selected_edge_list
            else:
                mylogger.info(f"Insert {len(filtered_global_candidate_edges)} Global New Edges")
                selected_edge_list += filtered_global_candidate_edges
        if len(filtered_other_candidate_edges) != 0 and mode == "diverse":
            if len(filtered_other_candidate_edges) + len(selected_edge_list) >= num_edge:
                mylogger.info(f"Insert {num_edge - len(selected_edge_list)} Local New Edges")
                random_idx = np.random.choice(len(filtered_other_candidate_edges), num_edge - len(selected_edge_list),
                                              replace=False)
                for idx in random_idx:
                    selected_edge_list.append(filtered_other_candidate_edges[idx])
                return selected_edge_list
            else:
                mylogger.info(f"Insert {len(filtered_other_candidate_edges)} Local New Edges")
                selected_edge_list += filtered_other_candidate_edges
        mylogger.info(f"Fail to find a new edge to insert, randomly choose {num_edge - len(selected_edge_list)}")
        random_idx = np.random.choice(len(filtered_all_candidate_edges), num_edge - len(selected_edge_list),
                                      replace=False)
        for idx in random_idx:
            selected_edge_list.append(filtered_all_candidate_edges[idx])
        return selected_edge_list

    @staticmethod
    def old_choose_new_layer(model, layer_names, covered_layers, max_num=10, mode="diverse"):
        """
        Given a DL model and candidate layer names list, select `num_layer` layers that have not been used in this layer
        Arguments:
            :model: the target model to be analyzed
            :layer_names: list of possible layer names
            :covered_layers: a list that stores all already covered layer types
            :max_num: the max number of layers to be found
            :mode:
                diverse -> first iterate global layers, then iterate local layers.
                random -> randomly choose layers.
        Return:
            The list of index layer to be chosen
        """
        existing_classes = ArchitectureUtils.get_all_layers_classes(model)
        global_candidate_layers = []
        other_candidate_layers = []
        for layer_name in layer_names:
            if layer_name not in covered_layers:
                global_candidate_layers.append(layer_name)
            else:
                other_candidate_layers.append(layer_name)
        idx = []
        print(f"Choosing Mutation Ratio Between 1 and {max_num}")
        num_layer = np.random.randint(1, max_num + 1)
        if len(global_candidate_layers) != 0 and mode == "diverse":
            if len(global_candidate_layers) >= num_layer:
                mylogger.info(f"Insert {num_layer} out of {len(global_candidate_layers)} Global New Layers")
                return list(np.random.choice(global_candidate_layers, num_layer, replace=False))
            else:
                mylogger.info(f"Insert {len(global_candidate_layers)} Global New Layers")
                idx = global_candidate_layers
        if len(other_candidate_layers) != 0 and mode == "diverse":
            if len(other_candidate_layers) + len(idx) >= num_layer:
                mylogger.info(f"Insert {num_layer - len(idx)} out of {len(other_candidate_layers)} Local New Layer")
                return idx + list(np.random.choice(other_candidate_layers, num_layer - len(idx), replace=False))
            else:
                idx += other_candidate_layers
        mylogger.info(f"Fail to find enough new layers to insert, randomly choose {num_layer - len(idx)}")
        return idx + list(np.random.choice(layer_names, num_layer - len(idx), replace=False))

    @staticmethod
    def choose_new_layer(model, layer_names, covered_layers, covered_edges, max_num=10, mode="diverse"):
        """
        Given a DL model and candidate layer names list, select `num_layer` layers that have not been used in this layer
        Arguments:
            :model: the target model to be analyzed
            :layer_names: list of possible layer names
            :covered_layers: a list that stores all already covered layer types
            :covered_edges: a list that stores all covered layer sequences
            :max_num: the max number of layers to be found
            :mode:
                diverse -> first iterate global layers, then iterate local layers.
                random -> randomly choose layers.
        Return:
            {layer.name: layer_class}
            The list of index layer to be chosen
        """
        from scripts.generation.layer_pools import LayerInputNdim
        num_layer = np.random.randint(1, max_num + 1)
        existing_classes = ArchitectureUtils.get_all_layers_classes(model)
        global_candidate_layer_sequence = defaultdict(list)  # {layer_right_class: [], layer_right_class: []}
        local_candidate_layer_sequence = defaultdict(list)  # {layer_right_class: [], layer_right_class: []}
        for layer_name in layer_names:
            for existing_layer in existing_classes:
                if existing_layer not in layer_names:
                    continue
                edge = (existing_layer, layer_name)
                if edge not in covered_edges:
                    if layer_name not in covered_layers:
                        global_candidate_layer_sequence[layer_name].append(existing_layer)
                    else:
                        local_candidate_layer_sequence[layer_name].append(existing_layer)

        def filter_sequences_by_ndim(candidate_sequence):
            """
            Given the candidate_sequence, choose the possible layer sequence that can be inserted.
            """
            filtered_sequences = defaultdict(list)
            for layer_class in candidate_sequence:
                required_input_ndim = LayerInputNdim[layer_class]
                possible_layer_list = ArchitectureUtils.scan_by_ndim(model, required_input_ndim)
                candidate_layer_class_list = candidate_sequence[layer_class]
                for layer in possible_layer_list:
                    if layer.__class__.__name__ in candidate_layer_class_list:
                        filtered_sequences[layer_class].append(layer)
            return filtered_sequences

        # filter layer sequences that cannot be directed inserted.
        filtered_global_layer_sequences = filter_sequences_by_ndim(
            global_candidate_layer_sequence)  # {layer_right_class: [layer1, layer2]}
        filtered_local_layer_sequences = filter_sequences_by_ndim(
            local_candidate_layer_sequence)  # {layer_right_class: [layer1, layer2]}

        selection_pair = {}
        if len(filtered_global_layer_sequences) != 0 and mode == "diverse":
            choose_num = min(num_layer, len(filtered_global_layer_sequences))
            mylogger.info(f"Insert {choose_num} out of {len(filtered_global_layer_sequences)} Global New Layers")
            right_layer_class_list = list(
                np.random.choice(list(filtered_global_layer_sequences.keys()), choose_num, replace=False))
            for right_layer_class in right_layer_class_list:
                left_layer = np.random.choice(filtered_global_layer_sequences[right_layer_class])
                selection_pair[left_layer.name] = right_layer_class
                mylogger.info('insert {} after {}'.format(right_layer_class, left_layer.name))
        if len(selection_pair) < num_layer and len(filtered_local_layer_sequences) != 0 and mode == "diverse":
            choose_num = min(num_layer - len(selection_pair), len(filtered_local_layer_sequences))
            mylogger.info(f"Insert {choose_num} out of {len(filtered_local_layer_sequences)} Local New Layers")
            right_layer_class_list = list(
                np.random.choice(list(filtered_local_layer_sequences.keys()), choose_num, replace=False))
            for right_layer_class in right_layer_class_list:
                left_layer = np.random.choice(filtered_local_layer_sequences[right_layer_class])
                selection_pair[left_layer.name] = right_layer_class
                mylogger.info('insert {} after {}'.format(right_layer_class, left_layer.name))
        if len(selection_pair) < num_layer:
            mylogger.info(
                f"Fail to find enough new layers to insert, randomly choose {num_layer - len(selection_pair)}")
            layer_class_list = list(np.random.choice(layer_names, num_layer - len(selection_pair), replace=False))
            for layer_class in layer_class_list:
                required_input_ndim = LayerInputNdim[layer_class]
                possible_layer_list = ArchitectureUtils.scan_by_ndim(model, required_input_ndim)
                if len(possible_layer_list) != 0:
                    # If we can find candidate layer, randomly select one
                    layer_index_to_insert = model.layers.index(np.random.choice(possible_layer_list))
                else:
                    # If no candidate layer can be found, we randomly choose enough amount of layers
                    mylogger.info(f"Fail to find suitable insertion point for dimension: {required_input_ndim}")
                    layer_index_to_insert = np.random.choice(np.arange(1, len(model.layers)))
                mylogger.info('insert {} after {}'.format(layer_class, model.layers[layer_index_to_insert].name))
                selection_pair[model.layers[layer_index_to_insert].name] = layer_class
        return selection_pair

    @staticmethod
    def scan_by_ndim(model, required_ndim):
        """
        Find the layer that can output the tensor with the required_ndim.
        Arguments:
            model: the DL model to be analyzed
            required_ndim: the required dimension (list)
        Return:
            The list of candidate layer.
        """
        if model.__class__.__name__ == 'Sequential':
            layer_list = model.layers
        else:
            layer_list = model.layers[1:]  # ignore the first input layer

        candidate_layer_list = []
        for layer in layer_list:
            for ndim in required_ndim:
                if len(layer.output_shape) == ndim:
                    candidate_layer_list.append(layer)
                    break
        return candidate_layer_list

    @staticmethod
    def get_all_layers_classes(model):
        if model.__class__.__name__ == 'Sequential':
            layer_list = model.layers
        else:
            layer_list = model.layers[1:]  # ignore the first input layer
        layer_class_set = set()
        for layer in layer_list:
            layer_class = layer.__class__.__name__
            layer_class_set.add(layer_class)
        return list(layer_class_set)

    @staticmethod
    def pick_layer_by_class(model, layer_class: str):
        """
        Pick a layer that belongs to a given class: `layer_class`.
        Arguments:
            :model: The model to be analyzed.
            :layer_class: The given layer class to be found.
        Return:
            The name of the searched layer.
        """
        layer_idx_list = []
        for i, layer in enumerate(model.layers):
            if layer.__class__.__name__ == layer_class:
                layer_idx_list.append(i)
        layer_idx = np.random.choice(layer_idx_list)
        return model.layers[layer_idx].name

    @staticmethod
    def find_layer_by_name(model, layer_name: str):
        """
        Find the layer by the given `layer_name`.
        Arguments:
            :model: The model to be analyzed.
            :layer_name: The given layer name to be picked.
        Return:
            The keras layer.
        """
        picked_layer = None
        for layer in model.layers:
            if layer.name == layer_name:
                picked_layer = layer
                return picked_layer
        # WARNING: If we enter this return, something wrong happens.
        return picked_layer

    @staticmethod
    def crop_tensor(tensor):
        """
        If a tensor has a very large shape, it may take a lot of time to do inference or even cause the OOM issue.
        This method is designed to check whether the tensor shape is indeed very large.
        And provide a layer sequence to crop it to a pre-defined range.
        Arguments:
            :tensor
        Return:
            Cropped tensor
        """
        cropped_tensor = tensor
        import keras
        dim = len(tensor.shape.as_list())
        if dim == 5:  # 5D
            r, c, t = tensor.shape.as_list()[1], tensor.shape.as_list()[2], tensor.shape.as_list()[3]
            if r is not None and c is not None and t is not None:
                if r > 20 or c > 20 or t > 20:
                    cropped_tensor = keras.layers.Cropping3D(
                        cropping=(
                            max(0, int((r - 10) / 2)),
                            max(0, int((c - 10) / 2)),
                            max(0, int((t - 10) / 2)),
                        )
                    )(tensor)
        elif dim == 4:  # 2D
            r, c = tensor.shape.as_list()[1], tensor.shape.as_list()[2]
            if r is not None and c is not None:
                if r > 20 or c > 20:
                    cropped_tensor = keras.layers.Cropping2D(
                        cropping=(
                            max(0, int((r - 10) / 2)),
                            max(0, int((c - 10) / 2)),
                        )
                    )(tensor)
        elif dim == 3:  # 1D
            t = tensor.shape.as_list()[1]
            if t is not None and t > 20:
                cropped_tensor = keras.layers.Cropping1D(cropping=int((t - 10) / 2))(tensor)
        return cropped_tensor

    @staticmethod
    def connect_two_layers(model, layer1_name, layer2_name):
        """
        Connect two layers.
        Arguments:
            :model: The DL model to be mutated.
            :layer1_name: The first layer.
            :layer2_name: The second layer.
        Return:
            The mutated DL model.
        """
        from scripts.mutation.mutation_utils import LayerUtils
        import keras
        input_layers = {}  # store the inputs of each layer in the DL model
        output_tensors = {}  # store the output of each layer in the DL model
        model_outputs = {}
        # Get the input of each layer in the DL model.
        for layer in model.layers:
            for node in layer._outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name not in input_layers.keys():
                    input_layers[layer_name] = [layer.name]
                else:
                    input_layers[layer_name].append(layer.name)

        output_tensors[model.layers[0].name] = model.input
        model_outputs[model.layers[0].name] = model.input
        if model.__class__.__name__ == 'Sequential':
            layer_list = model.layers
        else:
            layer_list = model.layers[1:]  # ignore the first Input layer

        # Iterate the whole DL model, during the iteration, we need to get the below information: for the edge l1 -> l2
        left_output = None  # the output of l1
        right_cloned_layer = None  # the cloned version of l2
        right_input_shape = None  # the dimension of the l2's input
        for layer in layer_list:
            layer_input_tensors = []
            for l in input_layers[layer.name]:
                layer_input_tensors.append(output_tensors[l])
                if l in model_outputs:
                    model_outputs.pop(
                        l)  # if the layer's output have been used, pop it out, otherwise consider it as the model's output

            if len(layer_input_tensors) == 1:
                layer_input_tensors = layer_input_tensors[0]
            if layer.name == layer1_name and layer1_name == layer2_name:
                # corner case when the layer1_name and layer2_name are the same
                cloned_layer = LayerUtils.clone(layer)
                x = cloned_layer(layer_input_tensors)
                left_output = x
                if int(keras.__version__.split(".")[1]) >= 7:
                    cloned_layer._name += "_1"
                else:
                    cloned_layer.name += "_1"
                right_input_shape = layer_input_tensors.shape.as_list()
                right_cloned_layer = LayerUtils.clone(layer)
                output_tensors[layer.name] = x
                model_outputs[layer.name] = x
                continue
            if layer.name == layer1_name:
                # get the left output
                cloned_layer = LayerUtils.clone(layer)
                x = cloned_layer(layer_input_tensors)
                left_output = x
            elif layer.name == layer2_name:
                # note that we need to clone the layer twice otherwise these two layers will require exactly the
                # same input shape
                cloned_layer = LayerUtils.clone(layer)
                x = cloned_layer(layer_input_tensors)
                if int(keras.__version__.split(".")[1]) >= 7:
                    cloned_layer._name += "_1"
                else:
                    cloned_layer.name += "_1"
                right_input_shape = layer_input_tensors.shape.as_list()
                right_cloned_layer = LayerUtils.clone(layer)
            else:
                cloned_layer = LayerUtils.clone(layer)
                x = cloned_layer(layer_input_tensors)
            output_tensors[layer.name] = x
            model_outputs[layer.name] = x

        # connect two layers
        # reshape the output tensor of left if the dimension is not matching
        from scripts.generation.layer_pools import LayerInputNdim
        right_class = right_cloned_layer.__class__.__name__
        if len(left_output.shape.as_list()) not in LayerInputNdim[right_class]:
            left_output = ArchitectureUtils._expand_drop_dim(left_output, right_input_shape)
        if int(keras.__version__.split(".")[1]) >= 7:
            right_cloned_layer._name += "_2"
        else:
            right_cloned_layer.name += "_2"
        edge_output = right_cloned_layer(left_output)
        # Crop the tensor to a limit range if the tensor's shape is too large.
        edge_output = ArchitectureUtils.crop_tensor(edge_output)
        return keras.Model(inputs=model.inputs, outputs=list(model_outputs.values()) + [edge_output])

    @staticmethod
    def find_two_layer_with_same_output_shape(model):
        """
        Find two layers that have the same output shape in the model.
        Arguments:
             :model: DL model to be analyzed
        Return:
            layer1_name: str
            layer2_name: str
        """
        output_shape_dict = {}
        if model.__class__.__name__ == 'Sequential':
            layer_list = model.layers
        else:
            layer_list = model.layers[1:]  # ignore the first input layer
        for layer in layer_list:
            output_shape = layer.output.shape.as_list()
            output_shape_str_list = [str(i) for i in output_shape[1:]]
            output_shape_str = "_".join(output_shape_str_list)
            if output_shape_str not in output_shape_dict:
                output_shape_dict[output_shape_str] = []
            output_shape_dict[output_shape_str].append(layer.name)
        filtered_shape_dict = {}
        for output_shape in output_shape_dict:
            if len(output_shape_dict[output_shape]) >= 2:
                filtered_shape_dict[output_shape] = output_shape_dict[output_shape]
        if len(filtered_shape_dict) == 0:
            return None, None
        selected_shape = np.random.choice(list(filtered_shape_dict.keys()))
        layer1_name, layer2_name = np.random.choice(filtered_shape_dict[selected_shape], 2, replace=False)
        return layer1_name, layer2_name

    @staticmethod
    def merge_two_layers(model, merge_layer, layer1_name, layer2_name):
        """
        Merge the output of layer1_name and layer2_name using merge_layer. The merged output will directly
        connect the model's output.
        Arguments:
            model: DL model to be mutated.
            merge_layer: the layer that is used to merge other two layers.
            layer1_name, layer2_name: the names of layers to be merged.
        Return:
            The mutated DL model.
        """
        from scripts.mutation.mutation_utils import LayerUtils
        import keras
        input_layers = {}
        output_tensors = {}
        model_outputs = {}
        for layer in model.layers:
            for node in layer._outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name not in input_layers.keys():
                    input_layers[layer_name] = [layer.name]
                else:
                    input_layers[layer_name].append(layer.name)

        output_tensors[model.layers[0].name] = model.input
        model_outputs[model.layers[0].name] = model.input
        left_output = None

        if model.__class__.__name__ == 'Sequential':
            layer_list = model.layers
        else:
            layer_list = model.layers[1:]  # ignore the first input layer
        for layer in layer_list:
            layer_input_tensors = []
            for l in input_layers[layer.name]:
                layer_input_tensors.append(output_tensors[l])
                if l in model_outputs:
                    model_outputs.pop(
                        l)  # if the layer's output have been used, pop it out, otherwise consider it as the model's output

            if len(layer_input_tensors) == 1:
                layer_input_tensors = layer_input_tensors[0]
            if layer.name in [layer1_name, layer2_name] and left_output is None:
                cloned_layer = LayerUtils.clone(layer)
                x = cloned_layer(layer_input_tensors)
                left_output = x
            elif layer.name in [layer1_name, layer2_name] and left_output is not None:
                # not that we need to clone the layer twice otherwise these two layers will require exactly the
                # same input shape
                cloned_layer = LayerUtils.clone(layer)
                if int(keras.__version__.split(".")[1]) >= 7:
                    cloned_layer._name += "_merge1"
                else:
                    cloned_layer.name += "_merge1"
                original_output = cloned_layer(layer_input_tensors)
                if int(keras.__version__.split(".")[1]) >= 7:
                    merge_layer._name += '_copy_' + "ML"
                else:
                    merge_layer.name += '_copy_' + "ML"
                x = merge_layer([original_output, left_output])
            else:
                cloned_layer = LayerUtils.clone(layer)
                x = cloned_layer(layer_input_tensors)
            output_tensors[layer.name] = x
            model_outputs[layer.name] = x
        import keras
        return keras.Model(inputs=model.inputs, outputs=list(model_outputs.values()))

    @staticmethod
    def _choose_config(config, layer_config, layer_pool):
        # Step 2: Search for another possible option for a config of the layer
        origin_param = layer_config[config]
        param_indices = np.arange(len(layer_pool[config]))
        idx = np.random.choice(param_indices)
        replace_result = layer_pool[config][idx]
        if replace_result == 0:
            # this means that we need to randomly generate some values
            if config == "filters" or config == "depth_multiplier" or config == "size" or config == "output_padding":
                replace_result = np.random.randint(1, 10)
            elif config == "kernel_size":
                dim = len(origin_param)
                replace_result = [np.random.randint(20) for i in range(dim)]
            elif config == "strides" or config == "dilation_rate" or config == "padding":
                dim = len(origin_param)
                replace_result = [np.random.randint(1, 5) for i in range(dim)]
            elif config == "pool_size" or config == "cropping":
                dim = len(origin_param)
                replace_result = [np.random.randint(10) for i in range(dim)]
            elif config == "units":
                replace_result = np.random.randint(10)
            elif config in ["dropout", "recurrent_dropout", "rate", "stddev",
                            "momentum", "epsilon", "l1", "l2", "alpha", "theta"
                                                                        "max_value", "negative_slope", "threshold"]:
                replace_result = float(np.random.rand(1))
            elif config in ["input_dim", "output_dim"]:
                replace_result = np.random.randint(100)
            elif config in ["n"]:
                replace_result = np.random.randint(1, 5)
        return replace_result

    @staticmethod
    def choose_layers_for_mparam(model, param_diversity, max_layer_num, max_config_num, mode, api_pool, mutable_config,
                                 numeric_param_size=5):
        """
        Randomly choose at most [max_num] layers to change the parameters of its layer input.
        Arguments:
            :model: The DL model to be analyzed.
            :param_diversity: {layer_class: {config1: [xxx], config2: [xxx], config3: [xxx]}, ...}
            :max_layer_num: At most [max_num] layers to be selected.
            :max_config_num: Mutate at most [max_config_num] configures in one layer.
            :mode:
                "diverse": we select uncovered layer shape for model to mutate.
                "random": we randomly pick layer and shape for model to mutate.
                "audee": mutate all mutable configs in one layer
        Return:
            selected_layer_config_pair: {
            layer1_name: {param1: value1, param2: value2}, layer2_name: {param1: value}, ...
            }
        """
        # Step 2.1: Find all possible layer classes that we can mutate their parameters
        layer_config_dict = {}
        layer_class_dict = {}
        for layer in model.layers:
            layer_class = layer.__class__.__name__
            layer_name = layer.name
            if layer_class not in api_pool:
                continue
            # Step 1.2 Filter out those layers that have no mutable configs.
            num_mutable_config = 0
            for config in api_pool[layer_class]:
                if config in mutable_config:
                    num_mutable_config += 1
                else:
                    pass
            if num_mutable_config != 0:
                layer_config_dict[layer_name] = layer.get_config()
                layer_class_dict[layer_name] = layer_class

        num_layer = np.random.randint(1, min(max_layer_num + 1, (len(layer_config_dict) + 1) // 2 + 1))

        # Randomly select [num_layer] layers
        selected_layer_list = np.random.choice(list(layer_config_dict.keys()), num_layer, replace=False)
        # Iterate all selected layers, find parameter value to mutate to.
        from collections import defaultdict
        selected_layer_config_pair = defaultdict(dict)
        for layer_name in selected_layer_list:
            # Step 3.1: Select possible parameter value for the given layer
            layer_class_name = layer_class_dict[layer_name]
            layer_config = layer_config_dict[layer_name]
            # Get the config space for the layer
            config_list = [config for config in api_pool[layer_class_name] if
                           config in mutable_config]
            np.random.shuffle(config_list)

            num_config = np.random.randint(1, max_config_num + 1)
            if len(config_list) < num_config:
                num_config = int(len(config_list) + 1 / 2)
            mylogger.info(
                f"Changing {num_config}/{len(config_list)} of layer {layer_name}'s configuration")

            total_change = 0  # record the total number of configs that have been mutated
            for selected_config in config_list:
                # Iterate through all mutable config, until we have mutated enough configs in this layer
                origin_param = layer_config[selected_config]
                if mode == "diverse":
                    if api_pool[layer_class_name][selected_config] == [0] and len(
                            param_diversity[layer_class_name][selected_config]) > numeric_param_size:
                        mylogger.info("Numeric Parameter Is Full!!")
                        continue
                    # repeatedly find the possible config that has never been used before
                    new_param = ArchitectureUtils._choose_config(selected_config, layer_config,
                                                                 api_pool[layer_class_name])
                    try_time = 0
                    while new_param in param_diversity[layer_class_name][selected_config]:
                        if try_time > 10:
                            break
                        new_param = ArchitectureUtils._choose_config(selected_config, layer_config,
                                                                     api_pool[layer_class_name])
                        try_time += 1
                else:
                    new_param = ArchitectureUtils._choose_config(selected_config, layer_config,
                                                                 api_pool[layer_class_name])
                selected_layer_config_pair[layer_name][selected_config] = new_param
                mylogger.info(
                    f"changing config {selected_config} in layer {layer_name}: from {origin_param} to {new_param}")
                total_change += 1
                if mode != "Audee" and total_change > num_config:
                    # if we are on Audee, we change all mutable config each time; Otherwise, we change at most 3 params
                    break
        return selected_layer_config_pair

    @staticmethod
    def generate_shape_by_ndims(input_ndim, min=None, max=None):
        def get_random():
            if min is None and max is None:
                return np.random.randint(10)
            else:
                return np.random.randint(min, max)

        input_shape = [None]
        input_shape += [get_random() for i in range(input_ndim - 1)]
        return input_shape

    @staticmethod
    def choose_layers_for_mshape(model, input_diversity, max_num, mode="diverse"):
        """
        Choose at most [max_num] layers to change the shape of its layer input.
        Arguments:
            :model: The DL model to be analyzed.
            :input_diversity: {layer_class: {"dtype": [xxx], "ndims": [xxx], "shape": [xxx]}, ...}
            :max_num: At most [max_num] layers to be selected.
            :mode:
                "diverse": we select uncovered layer shape for model to mutate.
                "random": we randomly pick layer and shape for model to mutate.
        Return:
            selected_layer_ndims_pair: {selected_layer_name: [selected_shape, origin_output_shape], ...}
        """

        from scripts.generation.layer_pools import LayerInputNdim
        from scripts.coverage.architecture_coverage import SHAPE_SPACE

        # Step 2.1: Find all possible layer classes that we can mutate their shape.
        global_layer_class_dict = {}
        local_layer_class_dict = {}
        all_layer_class_dict = {}
        all_layer_names = {}  # {layer_name: [input_shape, output_shape], ...}
        for layer in model.layers:
            layer_class = layer.__class__.__name__
            layer_name = layer.name
            if layer_class == "InputLayer" or layer_class not in LayerInputNdim:
                continue
            shape_list = input_diversity[layer_class]["shape"]
            if layer_class not in all_layer_class_dict:
                all_layer_class_dict[layer_class] = shape_list
            if len(input_diversity[layer_class]["shape"]) < SHAPE_SPACE:
                global_layer_class_dict[layer_class] = shape_list
            else:
                local_layer_class_dict[layer_class] = shape_list
            input_shape = list(layer.input.shape)
            output_shape = list(layer.output.shape)
            all_layer_names[layer_name] = [input_shape, output_shape]
        assert len(local_layer_class_dict) + len(global_layer_class_dict) == len(all_layer_class_dict)

        def find_num_layer_from_dict(num, layer_class_dict):
            if len(layer_class_dict) > num:
                mylogger.info(f"Changing {num} out of {len(layer_class_dict)} Layer's Shape.")
                layer_class_list = list(np.random.choice(list(layer_class_dict.keys()), num, replace=False))
                return layer_class_list, 0
            else:
                mylogger.info(f"Changing {len(layer_class_dict)} Layer's Shape.")
                layer_class_list = list(layer_class_dict.keys())
                return layer_class_list, num - len(layer_class_dict)

        num_layer = np.random.randint(1, min(max_num + 1, (len(all_layer_class_dict) + 1) // 2 + 1))
        mylogger.info(f"Change {num_layer} Layer's Shape Out Of {len(all_layer_class_dict)} Layer classes")
        selected_layer_pair = {}  # layer1_name: [shape, [input_shape, output_shape]]

        if len(global_layer_class_dict) > 0 and mode == "diverse":
            selected_layer_class_list, num_layer = find_num_layer_from_dict(num_layer, global_layer_class_dict)
            for layer_class in selected_layer_class_list:
                layer_name = ArchitectureUtils.pick_layer_by_class(model, layer_class)
                input_shape = all_layer_names[layer_name][0]
                output_shape = all_layer_names[layer_name][1]
                generated_shape = ArchitectureUtils.generate_shape_by_ndims(len(input_shape))
                while str(generated_shape) in global_layer_class_dict[layer_class]:
                    generated_shape = ArchitectureUtils.generate_shape_by_ndims(len(input_shape))
                selected_layer_pair[layer_name] = [generated_shape, output_shape]
                mylogger.info(
                    f"[Globally] Changing The Input Shape Of Layer: {layer_name} From {input_shape} To {generated_shape}")
        if num_layer != 0:
            if mode == "diverse":
                selected_layer_class_list, num_layer = find_num_layer_from_dict(num_layer, local_layer_class_dict)
            elif mode == "random":
                selected_layer_class_list, num_layer = find_num_layer_from_dict(num_layer, all_layer_class_dict)
            else:
                selected_layer_class_list = []
            for layer_class in selected_layer_class_list:
                layer_name = ArchitectureUtils.pick_layer_by_class(model, layer_class)
                input_shape = all_layer_names[layer_name][0]
                output_shape = all_layer_names[layer_name][1]
                generated_shape = ArchitectureUtils.generate_shape_by_ndims(len(input_shape))
                while generated_shape == input_shape:
                    generated_shape = ArchitectureUtils.generate_shape_by_ndims(len(input_shape))
                selected_layer_pair[layer_name] = [generated_shape, output_shape]
                mylogger.info(
                    f"[Locally] Changing The Input Shape Of Layer: {layer_name} From {input_shape} To {generated_shape}")
        return selected_layer_pair

    @staticmethod
    def choose_layers_for_mdtype(model, input_diversity, max_num, mode="diverse"):
        """
        Choose at most [max_num] layers to change the ndims of its layer input.
        Arguments:
            :model: The DL model to be analyzed.
            :input_diversity: {layer_class: {"dtype": [xxx], "ndims": [xxx], "shape": [xxx]}, ...}
            :max_num: At most [max_num] layers to be selected.
            :mode:
                "diverse": we select uncovered layer input dtype for model to mutate.
                "random": we randomly pick layer and layer input dtype that is different from original one for model to mutate.
        Return:
            selected_layer_dtype_pair: {selected_layer_name: selected_dtype, ...}
        """
        # Step 2.1.1: Set [num_layer] and make sure [num_layer] is smaller than the number of all mutable layer classes.
        from scripts.generation.layer_pools import LayerInputNdim
        global_layer_class_dict = {}
        local_layer_class_dict = {}
        all_layer_class_dict = {}
        for layer in model.layers:
            possible_dtype = {'bfloat16', 'double', 'float16', 'float32', 'float64', 'half'}
            layer_class = layer.__class__.__name__
            if layer_class == "InputLayer" or layer_class not in LayerInputNdim:
                continue
            # WORKAROUND: when mutating BN, we do not consider "float64" and "double" because it will lead to crash: https://github.com/keras-team/keras/issues/17044
            if layer_class == "BatchNormalization":
                possible_dtype = {'bfloat16', 'float16', 'float32', 'half'}
            if layer_class not in all_layer_class_dict:
                all_layer_class_dict[layer_class] = list(possible_dtype - {layer.input.dtype})
            current_dtype_set = set(input_diversity[layer_class]["dtype"])
            uncovered_dtype_set = possible_dtype - current_dtype_set
            if len(uncovered_dtype_set) > 0:
                global_layer_class_dict[layer_class] = list(uncovered_dtype_set)
            else:
                local_layer_class_dict[layer_class] = list(possible_dtype - {layer.input.dtype})
        assert len(local_layer_class_dict) + len(global_layer_class_dict) == len(all_layer_class_dict)

        def find_num_layer_from_dict(num, layer_class_dict):
            if len(layer_class_dict) > num:
                mylogger.info(f"Changing {num} out of {len(layer_class_dict)} Layer's DType.")
                layer_class_list = list(np.random.choice(list(layer_class_dict.keys()), num, replace=False))
                return layer_class_list, 0
            else:
                mylogger.info(f"Changing {len(layer_class_dict)} Layer's Dtype.")
                layer_class_list = list(layer_class_dict.keys())
                return layer_class_list, num - len(layer_class_dict)

        num_layer = np.random.randint(1, min(max_num + 1, (len(all_layer_class_dict) + 1) // 2 + 1))
        mylogger.info(f"Change {num_layer} Layer's DType Out Of {len(all_layer_class_dict)} Layer classes")
        selected_layer_pair = {}  # {layer1_name: dtype, layer2_name: dtype}
        if len(global_layer_class_dict) > 0 and mode == "diverse":
            selected_layer_class_list, num_layer = find_num_layer_from_dict(num_layer, global_layer_class_dict)
            for layer_class in selected_layer_class_list:
                layer_name = ArchitectureUtils.pick_layer_by_class(model, layer_class)
                selected_dtype = np.random.choice(global_layer_class_dict[layer_class])
                selected_layer_pair[layer_name] = selected_dtype
                mylogger.info(f"Selecting a Global DType {selected_dtype} For Layer {layer_name}")
        if num_layer != 0:
            if mode == "diverse":
                selected_layer_class_list, num_layer = find_num_layer_from_dict(num_layer, local_layer_class_dict)
            elif mode == "random":
                selected_layer_class_list, num_layer = find_num_layer_from_dict(num_layer, all_layer_class_dict)
            else:
                selected_layer_class_list = []
            for layer_class in selected_layer_class_list:
                layer_name = ArchitectureUtils.pick_layer_by_class(model, layer_class)
                selected_dtype = np.random.choice(global_layer_class_dict[layer_class])
                selected_layer_pair[layer_name] = selected_dtype
                mylogger.info(
                    f"No Global DType Can Be Found, Randomly Select DType: {selected_dtype} For Layer: {layer_name}")
        return selected_layer_pair

    @staticmethod
    def choose_layers_for_mdims(model, input_diversity, max_num, mode="diverse"):
        """
        Choose at most [max_num] layers to change the ndims of its layer input.
        Arguments:
            :model: The DL model to be analyzed.
            :input_diversity: {layer_class: {"dtype": [xxx], "ndims": [xxx], "shape": [xxx]}, ...}
            :max_num: At most [max_num] layers to be selected.
            :mode:
                "diverse": we select uncovered layer ndim for model to mutate.
                "random": we randomly pick layer and ndim for model to mutate.
        Return:
            selected_layer_ndims_pair: {selected_layer_name: [selected_ndim, origin_output_shape], ...}
        """
        # Step 2.1.1: Set [num_layer] and make sure [num_layer] is smaller than the number of all mutable layer classes.
        from scripts.generation.layer_pools import LayerInputNdim
        candidate_layer_class_list = ArchitectureUtils.find_dims_mutable_layers(model)
        if len(candidate_layer_class_list) == 0:
            raise ValueError("No Mutable Layers Can Be Found!!")
        num_layer = np.random.randint(1, min(max_num + 1, (len(candidate_layer_class_list) + 1) // 2 + 1))
        mylogger.info(f"Change {num_layer} Layer's Dimension Out Of {len(candidate_layer_class_list)} Layer classes")
        # Step 2.1.2: collect global_layer_class, local_layer_class, all_layer_class
        global_layer_class_dict = {}  # global_layer_class will record all possible layer's possible new ndims
        local_layer_class_dict = {}
        all_layer_class_dict = {}
        for layer_class in candidate_layer_class_list:
            current_ndims_set = set(input_diversity[layer_class]["ndims"])
            covered_ndims_set = current_ndims_set
            # if layer_class == "BatchNormalization":
            #     covered_ndims_set = current_ndims_set.union({2})  # WORKAROUND: when mutating BN, we do not consider 2D dimension, this is a potential bug.
            all_ndims_set = set(LayerInputNdim[layer_class])
            all_layer_class_dict[layer_class] = list(all_ndims_set)
            if len(all_ndims_set - covered_ndims_set) == 0:  # if we have iterated all possible ndims of all available layers
                local_layer_class_dict[layer_class] = list(all_ndims_set)
            else:
                global_layer_class_dict[layer_class] = list(all_ndims_set - covered_ndims_set)
        mylogger.info(f"Global Layer Class: {len(global_layer_class_dict)}; "
                      f"Local Layer Class: {len(local_layer_class_dict)}; "
                      f"Total Layer Class: {len(all_layer_class_dict)}")
        selected_layer_pair = {}  # {layer1_name: [new_ndim, origin_shape], layer2_name: ndim}

        def find_num_layer_from_dict(num, layer_class_dict):
            if len(layer_class_dict) > num:
                mylogger.info(f"Changing {num} out of {len(layer_class_dict)} Layer's NDims.")
                layer_class_list = list(np.random.choice(list(layer_class_dict.keys()), num, replace=False))
                return layer_class_list, 0
            else:
                mylogger.info(f"Changing {len(layer_class_dict)} Layer's NDims.")
                layer_class_list = list(layer_class_dict.keys())
                return layer_class_list, num - len(layer_class_dict)

        if len(global_layer_class_dict) >= 0 and mode == "diverse":
            selected_layer_class_list, num_layer = find_num_layer_from_dict(num_layer, global_layer_class_dict)
            for layer_class in selected_layer_class_list:
                selected_layer_name = ArchitectureUtils.pick_layer_by_class(model, layer_class)
                layer = ArchitectureUtils.find_layer_by_name(model, selected_layer_name)
                origin_output_shape = layer.output.shape
                selected_ndim = np.random.choice(global_layer_class_dict[layer_class])
                selected_layer_pair[selected_layer_name] = [selected_ndim, origin_output_shape]
                mylogger.info(f"Selecting a Global NDims {selected_ndim} For Layer {selected_layer_name}")
        if num_layer != 0:
            if mode == "diverse":
                selected_layer_class_list, num_layer = find_num_layer_from_dict(num_layer, local_layer_class_dict)
            elif mode == "random":
                selected_layer_class_list, num_layer = find_num_layer_from_dict(num_layer, all_layer_class_dict)
            else:
                selected_layer_class_list = []
            for layer_class in selected_layer_class_list:
                selected_layer_name = ArchitectureUtils.pick_layer_by_class(model, layer_class)
                layer = ArchitectureUtils.find_layer_by_name(model, selected_layer_name)
                origin_output_shape = layer.output.shape
                current_ndims = len(layer.input.shape)
                selected_ndim = np.random.choice(list(set(LayerInputNdim[layer_class]) - {current_ndims}))
                selected_layer_pair[selected_layer_name] = [selected_ndim, origin_output_shape]
                mylogger.info(
                    f"No Global NDims Can Be Found, Randomly Select NDims: {selected_ndim} For Layer: {selected_layer_name}")
        return selected_layer_pair

    @staticmethod
    def find_dims_mutable_layers(model):
        """
        Scan the model, find the layers that we can mutate the dimension of its input.
        Arguments:
            :model: The DL model to be analyzed.
        Return:
            A list of layer's class.
        """
        from scripts.generation.layer_pools import LayerInputNdim
        candidate_layer_class_list = []
        for layer in model.layers:
            layer_class = layer.__class__.__name__
            if layer_class not in LayerInputNdim:
                continue
            if len(LayerInputNdim[layer_class]) > 1 and layer_class not in candidate_layer_class_list:
                candidate_layer_class_list.append(layer_class)
        return candidate_layer_class_list

    @staticmethod
    def extract_edges(model):
        """

        Parameters
        ----------
        model

        Returns
        existing_edges: unique direct edges (tuple)
        -------

        """
        if model.__class__.__name__ == 'Sequential':
            layer_list = model.layers
        else:
            layer_list = model.layers[1:]  # ignore the first input layer

        existing_edges = []
        # collect existing edges and all layer classes
        for layer in layer_list:
            start_layer_class = layer.__class__.__name__
            for node in layer._outbound_nodes:
                end_layer_class = node.outbound_layer.__class__.__name__
                edge = (start_layer_class, end_layer_class)  # edge should be direct
                # Warning! I cannot remember why I write below two lines of code, I have now comment it.
                # Please pay attention to these three lines if meet some bugs.
                # edge_str = f"{start_layer_class}_{end_layer_class}"
                # if edge_str in global_edge_dict:
                #     global_edge_dict[edge_str] += 1
                if edge not in existing_edges:
                    existing_edges.append(edge)
        existing_edges = ArchitectureUtils.trim_edge(
            existing_edges)  # trim edge to remove edges that are not from the given layer.
        return existing_edges

    @staticmethod
    def extract_nodes(model):
        """

        Parameters
        ----------
        model

        Returns
        existing_nodes: {"layer_name1": [layer_config1, layer_config2], "layer_name2": [], ...}
        -------

        """
        if model.__class__.__name__ == 'Sequential':
            layer_list = model.layers
        else:
            layer_list = model.layers[1:]  # ignore the first input layer

        existing_nodes = {}

        # Check if the layer belongs to the target class.
        from scripts.generation.layer_pools import LAYERLIST
        total_layers = []
        for layer_type in LAYERLIST:
            target_layers = LAYERLIST[layer_type]
            total_layers += list(target_layers.available_layers.keys())
        # collect existing edges and all layer classes
        for layer in layer_list:
            layer_config = layer.get_config()
            layer_config.pop("name")
            if "filters" in layer_config: layer_config.pop("filters")
            if "units" in layer_config: layer_config.pop("units")
            layer_class = layer.__class__.__name__
            if layer_class not in total_layers:
                continue
            if layer_class not in existing_nodes:
                existing_nodes[layer_class] = []
            if layer_config not in existing_nodes[layer_class]:
                existing_nodes[layer_class].append(layer_config)
        return existing_nodes

    @staticmethod
    def extract_inputs(model):
        """

        Parameters
        ----------
        model

        Returns
        existing_inputs: {"layer_class": {"ndims": [], "dtype": [], "shape": []}}
        -------

        """
        if model.__class__.__name__ == 'Sequential':
            layer_list = model.layers
        else:
            layer_list = model.layers[1:]  # ignore the first input layer

        existing_inputs = {}
        # collect existing inputs and all layer classes
        for layer in layer_list:
            layer_class = layer.__class__.__name__
            from scripts.generation.layer_pools import LayerInputNdim
            if layer_class not in LayerInputNdim:
                # We skip those layers: merging layer.
                continue
            if layer_class not in existing_inputs:
                existing_inputs[layer_class] = {"ndims": [], "dtype": [], "shape": []}
            ndims = len(layer.input.shape)
            dtype = layer.input.dtype
            shape = str(list(layer.input.shape))  # we store the list to string because the list is unhashable for set
            if ndims not in existing_inputs[layer_class]['ndims']:
                existing_inputs[layer_class]['ndims'].append(ndims)
            if dtype not in existing_inputs[layer_class]['dtype']:
                existing_inputs[layer_class]['dtype'].append(dtype)
            if shape not in existing_inputs[layer_class]['shape']:
                existing_inputs[layer_class]['shape'].append(shape)
        return existing_inputs


if __name__ == "__main__":
    pass
