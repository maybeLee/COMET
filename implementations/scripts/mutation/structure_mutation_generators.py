from scripts.mutation.comet_mutation_operators import *
from scripts.mutation.structure_mutation_operators import WS_mut, GF_mut, NEB_mut, NAI_mut, NS_mut, ARem_mut, ARep_mut, LA_mut, LC_mut, LR_mut, LS_mut, MLAMut
import os
import gc

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
mylogger = Logger()

mlaMut = MLAMut()
MAX_LAYER_NUM = 10  # Hyper-parameter for mutation operators


def generate_model_by_model_mutation(model, operator, mutation_operator_mode=None,
                                     mutate_ratio=1, architecture_measure=None):
    """
    Generate models using specific mutate operator
    :param model: model loaded by keras (tensorflow backend default)
    :param operator: mutation operator
    :param mutate_ratio: ratio of selected neurons
    :return: mutation model object
    """
    # Baseline Mutation Operators
    if operator == 'WS':
        mutate_indices = utils.ModelUtils.weighted_layer_indices(model)
        mylogger.info("Generating model using {}".format(operator))
        return WS_mut(model=model, mutation_ratio=mutate_ratio, mutated_layer_indices=mutate_indices)
    elif operator == 'GF':
        mylogger.info("Generating model using {}".format(operator))
        return GF_mut(model=model, mutation_ratio=mutate_ratio)
    elif operator == 'NEB':
        mylogger.info("Generating model using {}".format(operator))
        return NEB_mut(model=model, mutation_ratio=mutate_ratio)
    elif operator == 'NAI':
        mylogger.info("Generating model using {}".format(operator))
        return NAI_mut(model=model, mutation_ratio=mutate_ratio)
    elif operator == 'NS':
        mylogger.info("Generating model using {}".format(operator))
        return NS_mut(model=model)
    elif operator == 'ARem':
        mylogger.info("Generating model using {}".format(operator))
        return ARem_mut(model=model)
    elif operator == 'ARep':
        mylogger.info("Generating model using {}".format(operator))
        return ARep_mut(model=model)
    elif operator == 'LA':
        mylogger.info("Generating model using {}".format(operator))
        return LA_mut(model=model)
    elif operator == 'LC':
        mylogger.info("Generating model using {}".format(operator))
        gc.collect()
        return LC_mut(model=model)
    elif operator == 'LR':
        mylogger.info("Generating model using {}".format(operator))
        gc.collect()
        return LR_mut(model=model)
    elif operator == 'LS':
        mylogger.info("Generating model using {}".format(operator))
        gc.collect()
        return LS_mut(model=model)
    elif operator == 'MLA':
        mylogger.info("Generating model using {}".format(operator))
        gc.collect()
        return mlaMut.mutate(model=model, mutated_layer_indices=None)
    # New Mutation Operators
    elif operator == "NLAll":
        mylogger.info("Generating model using {}".format(operator))
        return InteractionMutationUtils.insert_layers(model=model, layer_type=operator, max_num=MAX_LAYER_NUM, architecture_measure=architecture_measure, mutation_mode=mutation_operator_mode)
    elif operator == "LMerg":
        mylogger.info("Generating model using {}".format(operator))
        return InteractionMutationUtils.merge_layers(model=model, architecture_measure=architecture_measure, mutation_mode=mutation_operator_mode)
    elif operator == "Edge":
        mylogger.info("Generating model using {}".format(operator))
        return InteractionMutationUtils.connect_layers(model=model, architecture_measure=architecture_measure, max_num=MAX_LAYER_NUM, mutation_mode=mutation_operator_mode)
    elif operator == "MParam" or operator == "Audee":
        # set mutated_layer_indices to None instead of [mutated_layer_index] to randomly select a layer to mutate
        if operator == "Audee":
            mutation_operator_mode = "Audee"
        mylogger.info("Generating model using {}".format(operator))
        return ConfigurationMutationUtils.mutate_param(model=model, config_type=operator,
                                                       architecture_measure=architecture_measure, max_layer_num=MAX_LAYER_NUM, mutation_mode=mutation_operator_mode)
    elif operator == "SpecialI":
        mylogger.info("Generating model using {}".format(operator))
        return InputMutationUtils.convert_to_special(model=model)
    elif operator == "MDims":
        mylogger.info("Generating model using {}".format(operator))
        return InputMutationUtils.mutate_dims(model=model, architecture_measure=architecture_measure, max_num=MAX_LAYER_NUM, mutation_mode=mutation_operator_mode)
    elif operator == "MDtype":
        mylogger.info("Generating model using {}".format(operator))
        return InputMutationUtils.mutate_dtype(model=model, architecture_measure=architecture_measure, max_num=MAX_LAYER_NUM, mutation_mode=mutation_operator_mode)
    elif operator == "MShape":
        mylogger.info("Generating model using {}".format(operator))
        return InputMutationUtils.mutate_shape(model=model, architecture_measure=architecture_measure, max_num=MAX_LAYER_NUM, mutation_mode=mutation_operator_mode)
    else:
        mylogger.info("No such Mutation operator {}".format(operator))
        return None


def baseline_mutate_ops():
    return ["WS", "GF", "NEB", "NAI", "NS", "ARem", "ARep", "LA", "LC", "LR", "LS", "MLA", "Audee"]


def comet_mutate_ops():
    return ["SpecialI", "MDims", "MDtype", "LResh", "CParam", "LMerg", "NLAll", "Edge"]


if __name__ == '__main__':
    pass
