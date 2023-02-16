from scripts.tools.architecture_utils import ArchitectureUtils
import json
from itertools import combinations_with_replacement

from scripts.tools.architecture_utils import ArchitectureUtils

api_config_pool = json.load(open("boostraps/api_implementations/api_config_pool.json", "rb+"))
##### Hyper Parameter Tunning Part #####
SHAPE_SPACE = 5
# The chosen value for numeric parameter space is 5
PARAMETER_SPACE = 5


########################################


class ArchitectureMeasure(object):
    def __init__(self):
        self.node_diversity = {}  # {"layer_name1": [layer_config1, layer_config2], "layer_name2": [], ...}
        self.edge_diversity = []  # [(layer_class1, layer_class2), (layer_class1, layer_class3)]
        self.mutant_diversity = {}  # {"nodes": node_diversity, "edges": edge_diversity}
        self.input_diversity = {}  # {"layer_class1": {"ndims": [], "dtype": [], "shape": [], "value": []}, ...}

        # get the coverage space (trs)
        from scripts.generation.layer_pools import LAYERLIST, LayerInputNdim, LayerOutputNdim
        # get the layer type space
        self.total_layers = []
        for layer_type in LAYERLIST:
            target_layers = LAYERLIST[layer_type]
            self.total_layers += list(target_layers.available_layers.keys())

        # get the edge space
        self.total_edges = []

        def check_connectable(l1, l2):
            merge_layer_class = ["Concatenate", "Average", "Maximum", "Minimum", "Add", "Subtract", "Multiply", "Dot",
                                 "Dense"]
            if l1 in merge_layer_class or l2 in merge_layer_class:
                return False

            l1_o = set(LayerOutputNdim[l1])
            l2_i = set(LayerInputNdim[l2])
            return len(l1_o.intersection(l2_i)) != 0  # True: connectable, False: inconnectable

        for edge in combinations_with_replacement(self.total_layers, 2):
            # edge of two different layers
            s, e = edge
            if check_connectable(s, e):
                self.total_edges.append((s, e))
            if s != e and check_connectable(e, s):
                self.total_edges.append((e, s))

        # get the layer parameter space
        self.total_param = {}
        self.total_param_list = {}
        self.total_param_num = 0
        for layer_class in api_config_pool:
            self.total_param[layer_class] = 0
            self.total_param_list[layer_class] = {}
            for config in api_config_pool[layer_class]:
                self.total_param_list[layer_class][config] = []
                if api_config_pool[layer_class][config] == [0]:
                    self.total_param[layer_class] += PARAMETER_SPACE
                else:
                    self.total_param[layer_class] += len(api_config_pool[layer_class][config])
            self.total_param_num += self.total_param[layer_class]

        # get the layer input space (input dtype, input ndims, input shape)
        from scripts.generation.layer_pools import LayerInputNdim
        self.POSSIBLE_DTYPE = {'bfloat16', 'double', 'float16', 'float32', 'float64', 'half'}
        self.SHAPE_SPACE = 5
        self.total_ndims = {}
        self.total_dtype = {}
        self.total_shape = {}
        self.total_ndims_num = 0
        self.total_dtype_num = len(LayerInputNdim) * len(self.POSSIBLE_DTYPE)
        self.total_shape_num = len(LayerInputNdim) * self.SHAPE_SPACE
        for layer_class in LayerInputNdim:
            self.total_ndims[layer_class] = len(LayerInputNdim[layer_class])
            self.total_dtype[layer_class] = len(self.POSSIBLE_DTYPE)
            self.total_shape[layer_class] = self.SHAPE_SPACE  # we manually set the denominator of input shape to be 5.
            self.total_ndims_num += len(LayerInputNdim[layer_class])
        self.total_input_num = self.total_ndims_num + self.total_dtype_num + self.total_shape_num

    def get_architecture_diversity(self, model=None, mutant_name=None):
        edge_diversity = ArchitectureUtils.extract_edges(model)
        node_diversity = ArchitectureUtils.extract_nodes(model)
        input_diversity = ArchitectureUtils.extract_inputs(model)
        assert type(edge_diversity) == list
        assert type(node_diversity) == dict
        assert type(input_diversity) == dict
        self.mutant_diversity[mutant_name] = {"nodes": node_diversity, "edges": edge_diversity,
                                              "inputs": input_diversity}
        return edge_diversity, node_diversity, input_diversity

    def compare_edge(self, old, new, update=False):
        assert type(old) == type(new), print(type(old), type(new))
        assert type(old) == list
        new_edge = False
        for edge in new:
            if edge not in old:
                new_edge = True
                if update is True:
                    self.edge_diversity.append(edge)
                else:
                    print(f"Find New Edge: {edge}")
        return new_edge

    def compare_node(self, old, new, update=False):
        assert type(old) == type(new), print(type(old), type(new))
        assert type(old) == dict
        new_node = False
        for layer_class in new:
            if layer_class not in old:
                new_node = True
                if update is True:
                    self.node_diversity[layer_class] = new[layer_class]
                else:
                    print(f"Find New Layer Type: {layer_class}")
                continue
            for layer_config in new[layer_class]:
                if layer_config not in old[layer_class]:
                    new_node = True
                    if update is True:
                        self.node_diversity[layer_class].append(layer_config)
                    else:
                        print(f"Find New Layer Config: {layer_config}")
        return new_node

    def compare_input(self, old, new, update=False):
        """
        Compare the input diversity between old diversity and new diversity
        old, new: {"layer_class1": {"ndims": [], "dtype": [], "shape": []}, ...}
        Arguments:
            :old: old diversity state.
            :new: newly collected diversity.
        Return:
            new_input: boolean means whether new diversity has been found.
        """
        assert type(old) == type(new), print(type(old), type(new))
        assert type(old) == dict
        new_input = False
        for layer_class in new:
            new_ndims = set(new[layer_class]["ndims"])
            new_dtype = set(new[layer_class]["dtype"])
            new_shape = set(new[layer_class]["shape"])
            if layer_class not in old:
                new_input = True
                self.input_diversity[layer_class] = {"ndims": new_ndims, "dtype": new_dtype, "shape": new_shape}
            else:
                old_ndims = set(old[layer_class]["ndims"])
                old_dtype = set(old[layer_class]["dtype"])
                old_shape = set(old[layer_class]["shape"])
                if len(new_ndims - old_ndims) > 0:
                    new_input = True
                    if update is True:
                        self.input_diversity[layer_class]["ndims"] = set.union(old_ndims, new_ndims)
                    else:
                        print(f"Find New NDims: {new_ndims - old_ndims} For Layer: {layer_class}")
                if len(set(new_dtype) - set(old_dtype)) > 0:
                    new_input = True
                    if update is True:
                        self.input_diversity[layer_class]["dtype"] = set.union(old_dtype, new_dtype)
                    else:
                        print(f"Find New DType: {new_dtype - old_dtype} For Layer: {layer_class}")
                if len(old_shape) < SHAPE_SPACE and len(new_shape - old_shape) > 0:
                    new_input = True
                    if update is True:
                        self.input_diversity[layer_class]["shape"] = set.union(old_shape, new_shape)
                    else:
                        print(f"Find New Shape: {new_shape - old_shape} For Layer: {layer_class}")
        return new_input

    def compare_diversity(self, new_diversity):
        new_edges, new_nodes, new_inputs = new_diversity
        global_new_node = self.compare_node(self.node_diversity, new_nodes, update=True)
        global_new_edge = self.compare_edge(self.edge_diversity, new_edges, update=True)
        global_new_input = self.compare_input(self.input_diversity, new_inputs, update=True)
        return global_new_node, global_new_edge, global_new_input

    def get_current_api_params(self):
        for layer_class in self.node_diversity:
            if layer_class in api_config_pool:
                layer_config_list = self.node_diversity[layer_class]
                hp, param_list = self._layer_config_coverage(layer_config_list, layer_class)
                self.total_param_list[layer_class] = param_list  # {config: []}
        return self.total_param_list

    def api_coverage(self):
        print(f"The API Coverage Is: {len(self.node_diversity)}/{len(self.total_layers)}")
        return len(self.node_diversity) / len(self.total_layers)

    def input_coverage(self):
        """
        input_cov = ndim_cov + dtype_cov + shape_cov
        """
        covered_ndims = self.ndims_coverage()
        covered_dtype = self.dtype_coverage()
        covered_shape = self.shape_coverage()
        print(f"The NDims Coverage Is: {covered_ndims}/{self.total_ndims_num}")
        print(f"The DType Coverage Is: {covered_dtype}/{self.total_dtype_num}")
        print(f"The Shape Coverage Is: {covered_shape}/{self.total_shape_num}")
        print(f"The Input Coverage Is: {covered_ndims + covered_dtype + covered_shape}/{self.total_input_num}")
        input_cov = (covered_ndims + covered_dtype + covered_shape) / self.total_input_num
        ndims_cov = covered_ndims / self.total_ndims_num
        dtype_cov = covered_dtype / self.total_dtype_num
        shape_cov = covered_shape / self.total_shape_num
        return input_cov, ndims_cov, dtype_cov, shape_cov

    def ndims_coverage(self):
        """
        ndims_cov
        """
        covered_ndims_num = 0
        for layer_class in self.input_diversity:
            ndims_list = self.input_diversity[layer_class]["ndims"]
            covered_ndims_num += len(ndims_list)
        return covered_ndims_num

    def dtype_coverage(self):
        covered_dtype_num = 0
        for layer_class in self.input_diversity:
            dtype_list = self.input_diversity[layer_class]["dtype"]
            covered_dtype_num += len(dtype_list)
        return covered_dtype_num

    def shape_coverage(self):
        covered_shape_num = 0
        for layer_class in self.input_diversity:
            shape_list = self.input_diversity[layer_class]["shape"]
            covered_shape_num += min(len(shape_list),
                                     self.SHAPE_SPACE)  # if the total number of shape is larger that SHAPE_SPACE, we set it as 100%
        return covered_shape_num

    def api_pair_coverage(self):
        print(f"The API Pair Coverage Is: {len(self.edge_diversity)}/{len(self.total_edges)}")
        return len(self.edge_diversity) / len(self.total_edges)

    @staticmethod
    def _layer_config_coverage(layer_config_list, layer_class):
        """
        Calculate The Configuration Coverage On Specific Layer Class.
        Arguments:
            :layer_config_list: a list of layer configurations: [layer_config_1, layer_config_2].
            :layer_class (str): a string specify the class of layer under analysis.
        Return:
            hp: count of param_value.
            param_list: {param1: [value1, value2], ...}
        """
        config_pool = api_config_pool[layer_class]
        param_list = {}
        for param in config_pool:
            param_list[param] = []
        hp = 0
        # Journal Submitted Version is Below.
        for layer_config in layer_config_list:
            for param in layer_config:
                if param not in param_list:
                    continue
                if config_pool[param] == [0]:
                    if layer_config[param] not in param_list[param] and len(param_list[param]) <= PARAMETER_SPACE:
                        param_list[param].append(layer_config[param])
                        hp += 1
                else:
                    if layer_config[param] not in param_list[param]:
                        param_list[param].append(layer_config[param])
                        hp += 1
        return hp, param_list

    def config_coverage(self):
        total_hp = 0
        for layer_class in self.node_diversity:
            if layer_class in api_config_pool:
                layer_config_list = self.node_diversity[layer_class]
                hp, param_list = self._layer_config_coverage(layer_config_list, layer_class)
                total_hp += hp
        print(f"The Configuration Coverage is: {total_hp}/{self.total_param_num}")
        return total_hp / self.total_param_num

    def update_diversity(self, model, model_name):
        new_diversity = self.get_architecture_diversity(model, model_name)
        global_new_node, global_new_edge, global_new_input = self.compare_diversity(new_diversity)
        if global_new_node is True or global_new_edge is True or global_new_input:
            return 1
        else:
            return 0

    def coverage(self):
        api_pair_cov = self.api_pair_coverage()
        config_cov = self.config_coverage()
        input_cov, ndims_cov, dtype_cov, shape_cov = self.input_coverage()
        return api_pair_cov, config_cov, input_cov, ndims_cov, dtype_cov, shape_cov
