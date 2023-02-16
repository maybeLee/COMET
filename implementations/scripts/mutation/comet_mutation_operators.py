from scripts.tools import utils
import warnings
from scripts.logger.logger import Logger
import json
import numpy as np
import signal
from scripts.prediction.custom_objects import custom_objects


warnings.filterwarnings("ignore")
mylogger = Logger()
audee_config_pool = json.load(open("scripts/evaluation/audee_api_config.json", "rb+"))
audee_config_meta = json.load(open("scripts/evaluation/audee_config_meta.json", "rb+"))
api_config_meta = json.load(open("boostraps/api_implementations/config_metadata.json", "rb+"))
api_config_pool = json.load(open("boostraps/api_implementations/api_config_pool.json", "rb+"))


def handler(signum, frame):
    mylogger.info(f"TIMEOUT!!!")
    raise Exception("TIMEOUT!!")


signal.signal(signal.SIGALRM, handler)


def is_shape_equal(shape_1, shape_2):
    if len(shape_1) != len(shape_2):
        return False
    for s1, s2 in zip(shape_1[1:], shape_2[1:]):
        if s1.value != s2.value:
            return False
    return True


def NL_mut(model, selection_pair, selected_layer_name_list, config_pair=None, required_input_shape=None, revert_shape=True):
    """
    Utility function to insert several layers into a specific DL model
    Arguments:
        :model: model to be mutated
        :selection_pair: {layer_name_before_inserted: selected_layer_1, layer_name_2: selected_layer_2, ...}
        :selected_layer_name_list: the class name list of layers to be inserted
        :config_pair: {layer_name_before_inserted: layer_config_1, layer_name_2: layer_config_2, ...}
        :revert_shape: boolean.
            True -> we will revert the tensor's shape to original one after adding new layers.
            False -> we will not revert the tensor's shape
    Return:
        the mutated model
    """
    # Step 3.1: define the layer insertion function
    def new_layer_addition(x, layer):
        x = layer(x)
        layer_input = x
        if required_input_shape is not None:
            from scripts.tools.architecture_utils import ArchitectureUtils
            layer_input = ArchitectureUtils.reshape_tensor(tensor=layer_input, target_shape=required_input_shape,
                                                           mode="custom")
        output_shape = list(layer_input.shape)
        selected_layer = selection_pair[layer.name]
        if config_pair is not None and layer.name in config_pair.keys():
            layer_list = selected_layer(output_shape, config=config_pair[layer.name])
        else:
            layer_list = selected_layer(output_shape)
        for layer in layer_list:
            print(f"[DEBUG] Inserting layer: {layer}")
            y = layer(layer_input)
            layer_input = y
        if revert_shape is False:
            # if we do not revert the tensor's shape, this function is finished
            return layer_input
        # Implement shape/dimension conversion
        from scripts.tools.architecture_utils import ArchitectureUtils
        target_shape = output_shape
        mylogger.info(f"Converting output shape {layer_input.shape} to actual shape {target_shape}")
        layer_input = ArchitectureUtils.reshape_tensor(tensor=layer_input, target_shape=target_shape, mode="custom")
        return layer_input

    # Step 3.2: define the control flow of the layer insertion process
    operation = {}  # initiate the operation list, example: {layer_name: new_layer_addition}
    for layer_name_before_insertion in selection_pair:
        # go through each location
        operation[layer_name_before_insertion] = new_layer_addition
    new_model = utils.ModelUtils.functional_model_operation(model, operation=operation)
    return new_model


def copy_model_from_json(model_json):
    import copy
    model_json_copy = copy.deepcopy(model_json)
    import keras
    model_copy = keras.models.model_from_json(model_json_copy, custom_objects=custom_objects())
    return model_copy


class InteractionMutationUtils:

    @staticmethod
    def insert_layers(model, layer_type, architecture_measure, max_num=10, mutation_mode="diverse"):
        """
        Insert a few layers to a seed model, different layers will be inserted into different locations.
        Arguments:
            :model: The DL model to be mutated
            :layer_type: the type of layer: NLAll, One Layer Option
            :max_num: insert [1, max_num) layers
            :mutated_layer_indices: Default value: None;
            :mutation_mode:
                diverse -> first select the global option, if no global option is available, randomly choose one.
                random -> randomly choose one option.
        Return:
            mutated model
        """
        NL_model = utils.ModelUtils.model_copy(model, layer_type)
        # Step 1: Get the layers to be inserted, layer_list
        from scripts.generation.layer_pools import LAYERLIST, LayerInputNdim
        from scripts.tools.architecture_utils import ArchitectureUtils
        if layer_type == "NLAll":
            layer_names, layer_list = [], []
            layer_type_list = ["LConv", "LPool", "LRnn", "LNorm", "LReg", "LAct", "LResh", "LLocal"]
            for layer_ty in layer_type_list:
                target_layers = LAYERLIST[layer_ty]
                layer_names += list(target_layers.available_layers.keys())
                layer_list += list(target_layers.available_layers.values())
        else:
            target_layers = LAYERLIST[layer_type]
            layer_names = list(target_layers.available_layers.keys())
            layer_list = list(target_layers.available_layers.values())
        covered_edges = architecture_measure.edge_diversity
        covered_layers = list(architecture_measure.node_diversity.keys())
        selection_pair = ArchitectureUtils.choose_new_layer(NL_model, layer_names, covered_layers, covered_edges, max_num=max_num, mode=mutation_mode)
        for layer_name, layer_class in selection_pair.items():
            selected_idx = layer_names.index(layer_class)
            selected_layer = layer_list[selected_idx]
            selection_pair[layer_name] = selected_layer
        # Step 3: Execute the mutation function
        mutated_model = NL_mut(NL_model, selection_pair, [])
        # Step 4: Return the mutated model.
        return mutated_model

    @staticmethod
    def merge_layers(model, architecture_measure, mutation_mode="diverse"):
        """
        Randomly choose two layers that have the same output shape, then merge them together.
        Arguments:
            :model: DL model to be mutated
            :mutation_mode:
                diverse -> first select the global option, if no global option is available, randomly choose one.
                random -> randomly choose one option.
        Return:
            mutated DL model
        """
        ML_model = utils.ModelUtils.model_copy(model, "LMerg")
        # Step 1: get all possible merging layers, then choose one that have not been seen in this model.
        from scripts.generation.layer_pools import LAYERLIST
        from scripts.tools.architecture_utils import ArchitectureUtils
        target_layers = LAYERLIST["LMerg"]
        layer_names = list(target_layers.available_layers.keys())
        layer_list = list(target_layers.available_layers.values())
        covered_layers = list(architecture_measure.node_diversity.keys())
        selected_layer_name_list = ArchitectureUtils.old_choose_new_layer(ML_model,
                                                                      layer_names,
                                                                      covered_layers,
                                                                      max_num=2,
                                                                      mode=mutation_mode)  # each time, we only conduct merging once...
        if len(selected_layer_name_list) == 0:
            raise ValueError("No Mutable Layers Can Be Found!!")
        assert len(selected_layer_name_list) == 1
        selected_layer_name = selected_layer_name_list[0]
        selected_idx = layer_names.index(selected_layer_name)
        selected_layer = layer_list[selected_idx]
        selected_layer_name = layer_names[selected_idx]
        # Step 2: find two layers that have the same output shape
        layer1_name, layer2_name = ArchitectureUtils.find_two_layer_with_same_output_shape(ML_model)
        if layer1_name is None or layer2_name is None:
            raise ValueError("Failed to add the merging layer, no suitable tensor to find")
        mylogger.info(f"Trying Merge Layers: {layer1_name} and {layer2_name} by {selected_layer_name}")
        # Step 3: merge the output of two layers.
        new_model = ArchitectureUtils.merge_two_layers(ML_model, selected_layer, layer1_name, layer2_name)
        mylogger.info(f"Success on Merge Layers: {layer1_name} and {layer2_name} by {selected_layer_name}")
        return new_model

    @staticmethod
    def connect_layers(model, architecture_measure, max_num=10, mutation_mode="diverse"):
        """
        Connect two layers.
        Arguments:
            :model: the DL model to be mutated
            :mutation_mode:
                diverse -> first select the global option, if no global option is available, randomly choose one.
                random -> randomly choose one option.
        Return:
            The mutated DL model
        """
        Edge_model = utils.ModelUtils.model_copy(model, "Edge")
        # Step 1: find all candidate edges (layer1, layer2) that have not been explored
        from scripts.tools.architecture_utils import ArchitectureUtils
        # we use `num_edge` to decide how many edges should we find
        covered_edges = architecture_measure.edge_diversity
        candidate_edge_list = ArchitectureUtils.find_new_edges(Edge_model, covered_edges, max_num=max_num, mode=mutation_mode)
        for candidate_edge in candidate_edge_list:
            mylogger.info(f"Candidate Edge: {candidate_edge}")
            Edge_model_json = Edge_model.to_json()
            del Edge_model
            Edge_model = copy_model_from_json(Edge_model_json)
            candidate_edge = list(candidate_edge)
            layer1_class, layer2_class = candidate_edge[0], candidate_edge[1]
            layer1_name = ArchitectureUtils.pick_layer_by_class(Edge_model, layer1_class)
            layer2_name = ArchitectureUtils.pick_layer_by_class(Edge_model, layer2_class)
            # Note that the edge should be: layer1_name -> layer2_name
            mylogger.info(f"Trying Adding Edge: {candidate_edge} by Connecting {layer1_name} and {layer2_name}")
            Edge_model = ArchitectureUtils.connect_two_layers(Edge_model, layer1_name, layer2_name)
            mylogger.info(f"Successfully Add Edge: {candidate_edge} by Connecting {layer1_name} and {layer2_name}")
        return Edge_model


class ConfigurationMutationUtils:

    @staticmethod
    def mutate_param(model, config_type, architecture_measure, max_layer_num=10, max_config_num=3, mutation_mode="diverse"):
        """
        Function for changing model's layer configuration
        This function should be a wrapper function.
        The core implementation should be implemented in the _change_config method
        Arguments:
        ----------
        :model: model to be mutated
        :config_type: the type of target config pool: MParam, Audee
        :architecture_measure: the architecture coverage class
        :max_num:  the max number of layers to be mutated.
        :mutation_mode:
            diverse -> first select the global option, if no global option is available, randomly choose one.
            random -> randomly choose one option.
        -------
        Return:
            The mutated DL model.
        """
        # Step 1: Copy the model
        MParam_model = utils.ModelUtils.model_copy(model, "MParam")
        # Step 2: Choose at most [max_layer_num] and mutate at most [max_config_num] for each selected layer
        # Step 2.1: config specific option (max_layer_num, mutable_config, api_pool) based on the config_type: "Audee" or "MParam"
        from scripts.tools.architecture_utils import ArchitectureUtils
        api_pool_dict = {"Audee": audee_config_pool, "MParam": api_config_pool}
        # Set the hyper-parameter for different mode: Audee, CParam
        if config_type == "Audee":
            max_layer_num = 1
            config_meta = audee_config_meta
        elif config_type == "MParam":
            config_meta = api_config_meta  # meta information for API's configuration
        else:
            raise ValueError(f"The configuration type: {config_type} is not implemented.")
        mutable_config = config_meta[config_type]

        # Step 2.2: Select layers, configs, and choose parameter values.
        from scripts.coverage.architecture_coverage import PARAMETER_SPACE
        selected_layer_params_pair = ArchitectureUtils.choose_layers_for_mparam(
            MParam_model,
            architecture_measure.get_current_api_params(),
            max_layer_num=max_layer_num,
            max_config_num=max_config_num,
            mode=mutation_mode,
            api_pool=api_pool_dict[config_type],
            mutable_config=mutable_config,
            numeric_param_size=PARAMETER_SPACE
        )
        # selected_layer_params_pair: {layer1_name: {param1: value1, param2: value2}, ...}

        # Step 3: Apply the mutated configure to the model
        def change_layer_config(x, layer):
            mutate_param_dict = selected_layer_params_pair[layer.name]
            origin_config = layer.get_config()
            new_config = {}
            for param_name in origin_config:
                if param_name in mutate_param_dict:
                    new_config[param_name] = mutate_param_dict[param_name]
                else:
                    new_config[param_name] = origin_config[param_name]
            layer = layer.from_config(new_config)
            x = layer(x)
            layer_input = x
            return layer_input

        operation = {}
        for layer_name in selected_layer_params_pair.keys():
            operation[layer_name] = change_layer_config
        new_model = utils.ModelUtils.functional_model_operation(MParam_model, operation=operation)

        return new_model


class InputMutationUtils:

    @staticmethod
    def convert_to_special(model, mutated_layer_indices=None):
        """
        Mutation Function Which Will Insert A Lambda Layer To Convert The Tensor Value To NaN/Inf/0.
        Arguments:
            :model: the seed model to be mutated
            :mutated_layer_indices: the location of the layer to be mutated
        Return:
            The new model
        """
        # Step 1: Copy the model
        SpecialI_model= utils.ModelUtils.model_copy(model, "SpecialI")
        # Step 2: Choose the location of model to be inserted.
        layer_index_to_insert = np.random.choice(
            len(SpecialI_model.layers)) if mutated_layer_indices is None else np.random.choice(mutated_layer_indices)
        special_list = [np.NaN, np.Inf, 0]
        target_value = np.random.choice(special_list)
        mylogger.info(f"Change Input Of Layer: {SpecialI_model.layers[layer_index_to_insert].name} To: {target_value}")

        # Step 3: Write mutation function.
        def special_input_mutation(x, layer):
            import keras
            x = layer(x)
            layer_input = keras.layers.Lambda(lambda x: x * target_value)(x)
            return layer_input

        # Step 4: Apply mutation function.
        new_model = utils.ModelUtils.functional_model_operation(SpecialI_model, operation={
            SpecialI_model.layers[layer_index_to_insert].name: special_input_mutation})
        return new_model

    @staticmethod
    def mdims(model, selected_layer_ndims_pair, revert_shape):
        """
        Low-level mutation operator to mutate dimensions.
        Arguments:
            :model: model to be mutated
            :selected_layer_ndims_pair: None
            :revert_shape (default: True):
                True -> reverts the dims to original dims.
                False -> does not revert.
        """
        from scripts.tools.architecture_utils import ArchitectureUtils

        # Step 3: Write mutation function.
        def mdims_layer_addition(x, layer):
            selected_ndim = selected_layer_ndims_pair[layer.name][0]
            origin_output_shape = selected_layer_ndims_pair[layer.name][1]
            target_shape = [None for i in range(selected_ndim)]
            x = ArchitectureUtils._expand_drop_dim(x, target_shape)
            layer_input = layer(x)
            if revert_shape is True:
                layer_input = ArchitectureUtils.reshape_tensor(layer_input, origin_output_shape)
            return layer_input

        # Step 4: Apply mutation function.
        operation = {}
        for layer_name_before_insertion in selected_layer_ndims_pair:
            operation[layer_name_before_insertion] = mdims_layer_addition
        new_model = utils.ModelUtils.functional_model_operation(model, operation=operation)
        return new_model

    @staticmethod
    def mutate_dims(model, max_num=10, architecture_measure=None, mutation_mode="diverse"):
        """
        Mutation Function Which Will Iterate The Possible Dimension Of [num_layer] Layer.
        Arguments:
            :model: the seed model to be mutated.
            :max_num: the [max_num] layers to be mutated.
            :architecture_measure: store the covered test space.
            :mutation_mode:
                diverse -> first select the global option, if no global option is available, randomly choose one.
                random -> randomly choose one option.
        Return:
            The new model.
        """
        # Step 1: Copy the model
        MDims_model = utils.ModelUtils.model_copy(model, "MDims")
        # Step 2: Choose the location of model to be inserted.
        from scripts.tools.architecture_utils import ArchitectureUtils
        # Step 2.1: Find all possible layer classes that we can mutate their dimension.
        selected_layer_ndims_pair = ArchitectureUtils.choose_layers_for_mdims(
            MDims_model,
            architecture_measure.input_diversity,
            max_num,
            mode=mutation_mode
        )
        new_model = InputMutationUtils.mdims(MDims_model, selected_layer_ndims_pair, revert_shape=True)
        return new_model

    @staticmethod
    def mutate_dtype(model, max_num=10, architecture_measure=None, mutation_mode="diverse"):
        """
        Mutation Function Which Will Iterate The Possible DType Of [num_layer] Layer.
        Arguments:
            :model: the seed model to be mutated.
            :max_num: the [max_num] layers to be mutated.
            :architecture_measure: record the dtype coverage of each layer's input
            :mutation_mode:
                diverse -> first select the global option, if no global option is available, randomly choose one.
                random -> randomly choose one option.
        Return:
            The new model.
        """
        from scripts.generation.custom_layers import CustomCastLayer
        from scripts.tools.architecture_utils import ArchitectureUtils
        # Step 1: Copy the model
        MDtype_model = utils.ModelUtils.model_copy(model, "MDtype")
        # Step 2: Choose the location of model to be inserted.
        selected_layer_dtype_pair = ArchitectureUtils.choose_layers_for_mdtype(
            MDtype_model,
            architecture_measure.input_diversity,
            max_num,
            mode=mutation_mode
        )

        # Step 3: Write mutation function.
        def mdtype_mutation(x, layer):
            selected_dtype = selected_layer_dtype_pair[layer.name]
            import keras
            origin_dtype = x.dtype.name
            x = CustomCastLayer(target_dtype=selected_dtype)(x)
            if int(keras.__version__.split(".")[1]) >= 7:
                layer._dtype = selected_dtype
            else:
                layer.dtype = selected_dtype
            layer_input = layer(x)
            layer_input = CustomCastLayer(target_dtype=origin_dtype)(layer_input)
            return layer_input

        # Step 4: Apply mutation function.
        operation = {}
        for layer_name_before_insertion in selected_layer_dtype_pair:
            operation[layer_name_before_insertion] = mdtype_mutation

        # Step 4: Apply mutation function.
        new_model = utils.ModelUtils.functional_model_operation(MDtype_model, operation=operation)
        return new_model

    @staticmethod
    def mshape(model, selected_layer_shape_pair, revert_shape):
        """
        Low-level mutation operator to mutate shape.
        Arguments:
            :model: model to be mutated
            :selected_layer_shape_pair: None
            :revert_shape (default: True):
                True -> reverts the dims to original dims.
                False -> does not revert.
        """
        from scripts.tools.architecture_utils import ArchitectureUtils

        # Step 3: Write mutation function.
        def mshape_layer_mutation(x, layer):
            generated_shape = selected_layer_shape_pair[layer.name][0]
            origin_output_shape = selected_layer_shape_pair[layer.name][1]
            x = ArchitectureUtils.reshape_tensor(x, generated_shape)
            layer_input = layer(x)
            if revert_shape is True:
                layer_input = ArchitectureUtils.reshape_tensor(layer_input, origin_output_shape)
            return layer_input

        # Step 4: Apply mutation function.
        operation = {}
        for layer_name_before_insertion in selected_layer_shape_pair:
            operation[layer_name_before_insertion] = mshape_layer_mutation
        new_model = utils.ModelUtils.functional_model_operation(model, operation=operation)
        return new_model

    @staticmethod
    def mutate_shape(model, max_num=10, architecture_measure=None, mutation_mode="diverse"):
        """
        Mutation Function Which Will Randomly Change The Input Shape Of Randomly Chosen Layer.
        Arguments:
            :model: the seed model to be mutated.
            :max_num: the [max_num] layers to be mutated.
            :architecture_measure: record the dtype coverage of each layer's input
            :mutation_mode:
                diverse -> first select the global option, if no global option is available, randomly choose one.
                random -> randomly choose one option.
        Return:
            The new model.
        """
        # Step 1: Copy the model
        MShape_model = utils.ModelUtils.model_copy(model, "MShape")
        # Step 2: Choose the location of model to be inserted.
        from scripts.tools.architecture_utils import ArchitectureUtils
        selected_layer_shape_pair = ArchitectureUtils.choose_layers_for_mshape(
            MShape_model,
            architecture_measure.input_diversity,
            max_num,
            mode=mutation_mode
        )
        new_model = InputMutationUtils.mshape(MShape_model, selected_layer_shape_pair, revert_shape=True)
        return new_model


if __name__ == '__main__':
    pass
