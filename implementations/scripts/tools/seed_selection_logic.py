import numpy as np
from scripts.logger.logger import Logger
from scripts.prediction.custom_objects import custom_objects
from utils.utils import get_files

main_logger = Logger()


def deprecated(func):
    """
    This is a decorator for deprecated functions.
    """
    def wrapper(*args, **kwargs):
        print(f"[Warning] Function '{func.__name__}' is deprecated.")
        return func(*args, **kwargs)
    return wrapper


def log_seed(items: list, item_type: str):
    main_logger.info(f"Logging for {item_type}")
    for i, item in enumerate(items):
        main_logger.info(f"Score for {item.name} is: {item.score}")


class Architecture:
    # atomic class for seed
    def __init__(self, name, config, pool_size=3):
        self.name = name
        self.config = config  # the config will be string instead of json dict
        self.config_list = [config]
        self.inconsistency = 0
        self.pool_size = pool_size  # one architecture will contain at most [pool_size] configs

    def check_inconsistency(self, inconsistency):
        if self.inconsistency < 10 <= inconsistency:  # 10 is the pre-defined threshold
            main_logger.info(f"New inconsistency issue found!!!")

    def update_inconsistency(self, inconsistency):
        self.inconsistency = inconsistency

    @staticmethod
    def config_to_architecture(config):
        """
        Convert config to model's architecture
        Parameters
        ----------
        config

        Returns
        -------

        """
        pass

    def choose_config(self, ):
        np.random.shuffle(self.config_list)
        main_logger.info(f"Randomly Choose One Config From the Config Pool: {len(self.config_list)}")
        return self.config_list[-1]


class Roulette:
    # Roulette algorithm for selecting seeds

    class Mutator:
        # atomic class for seed
        def __init__(self, name, selected=0):
            self.name = name
            self.selected = selected
            self.reward_sum = 1.0

        @property
        def score(self):
            # fitness function will determine the score
            return self.reward_sum / (self.selected + 1)

        def increase_selected(self, ):
            self.selected += 1

        def increase_reward(self, ):
            self.reward_sum += 1

    def __init__(self, seed_names=None, capacity=61):
        self.capacity = capacity
        if seed_names is None:
            self._seeds = []
        else:
            self._seeds = [Roulette.Mutator(name) for name in seed_names]

    @property
    def seeds(self):
        seeds = {}
        for seed in self._seeds:
            seeds[seed.name] = seed
        return seeds

    def sort_seeds(self, ):
        # sort seed by its score
        import random
        random.shuffle(self._seeds)
        self._seeds.sort(key=lambda seed: seed.score, reverse=True)

    def pop_worst_seed(self, ):
        self.sort_seeds()
        self._seeds.pop()

    def is_full(self):
        if len(self._seeds) > self.capacity:
            return True
        else:
            return False

    def choose_seed(self, num_seed=1):
        # choose seed based on its score: intuitively speaking, we intend to select seed that has higher score
        sum = 0
        for seed in self._seeds:
            sum += seed.score
        rand_num = np.random.rand() * sum
        for seed in self._seeds:
            if rand_num < seed.score:
                return seed.name
            else:
                rand_num -= seed.score

    def increase_selected(self, seed_name):
        self.seeds[seed_name].increase_selected()

    def increase_reward(self, seed_name):
        self.seeds[seed_name].increase_reward()


class MutatorSelector:
    class MutatorSeed:
        def __init__(self, name):
            self.name = name
            self.reward_sum = 1
            self.selected = 0

        def increase_reward(self, ):
            self.reward_sum += 1

        def increase_reward_by_value(self, value):
            self.reward_sum += value

        def increase_selected(self, ):
            self.selected += 1

        @property
        def score(self):
            return self.reward_sum / (self.selected+1)

    def __init__(self, initial_seed_mode=None, selection_mode=None, mutation_operator_list=None, architecture_pool=50):
        from scripts.mutation.structure_mutation_generators import baseline_mutate_ops
        if mutation_operator_list == "old":
            self._mutators = baseline_mutate_ops()
        else:
            self._mutators = mutation_operator_list.split(" ")
        if initial_seed_mode == "synthesis":
            architecture_names, architecture_configs = self._load_initial_config("/root/data/synthesized_models/")  # replace me
        elif initial_seed_mode == "origin":
            architecture_names, architecture_configs = self._load_initial_model("/root/data/origin_models/")  # replace me
        elif initial_seed_mode == "test":
            architecture_names, architecture_configs = self._load_initial_config("/root/data/test_models/")  # replace me
        else:
            raise NotImplementedError(f"Initial Seed Mode: {initial_seed_mode} It Not Implemented")

        self.architectures = [Architecture(name=name, config=config, pool_size=1) for name, config in zip(architecture_names, architecture_configs)]
        self.initial_architectures = self.architectures
        self.architecture_pool = architecture_pool
        self._seeds = []
        self.mcmc = None
        self._original_seeds_name = []
        self.selection_mode = selection_mode
        self.last_used_seed = None

    @staticmethod
    def _load_initial_config(initial_seed_dir, initial_seed_mode=None):
        # load initial seeds' configuration from the initial_seed_dir
        architecture_config_path_list = get_files(initial_seed_dir, ".json")
        architecture_configs = []
        architecture_names = []
        for architecture_path in architecture_config_path_list:
            architecture_name = architecture_path.split("/")[-1].split(".json")[0]
            if initial_seed_mode is not None:
                # we have specifically set some initial seeds
                initial_seed_list = initial_seed_mode.split(" ")
                if architecture_name not in initial_seed_list:
                    continue
            architecture_names.append(architecture_name)
            model = None
            with open(architecture_path, "r") as file:
                architecture_config = file.read()
                import keras

                model = keras.models.model_from_json(architecture_config, custom_objects=custom_objects())
            x = model.inputs
            y = model.outputs
            import keras
            new_model = keras.Model(x, y)  # to avoid some configuration that does not have InputLayer
            if int(keras.__version__.split(".")[1]) >= 7:
                new_model._name = architecture_name
            else:
                new_model.name = architecture_name
            architecture_configs.append(new_model.to_json())
            del new_model
            del model
            from keras import backend as K
            K.clear_session()
        return architecture_names, architecture_configs

    @staticmethod
    def _load_initial_model(initial_model_dir):
        # load initial model (large ones) from the initial_model_dir
        model_path_list = get_files(initial_model_dir, ".h5")
        architecture_configs = []
        architecture_names = []
        for model_path in model_path_list:
            architecture_name = model_path.split("/")[-1].rstrip(".h5")
            import tensorflow as tf
            print(f"TensorFlow is using GPU? {tf.test.gpu_device_name()}")
            print(f"Loading model: {architecture_name}")
            architecture_names.append(architecture_name)
            model = None
            import keras
            model = keras.models.load_model(model_path, custom_objects=custom_objects())
            x = model.inputs
            y = model.outputs
            new_model = keras.Model(x, y)  # to avoid some configuration that does not have InputLayer
            if int(keras.__version__.split(".")[1]) >= 7:
                new_model._name = architecture_name
            else:
                new_model.name = architecture_name
            architecture_configs.append(new_model.to_json())
            del new_model
            del model
            from keras import backend as K
            K.clear_session()
        return architecture_names, architecture_configs

    # Utils
    def pop_architecture_by_name(self, popped_name):
        new_architecture_list = []
        popped_architecture = None
        for architecture in self.architectures:
            if architecture.name == popped_name:
                popped_architecture = architecture
            else:
                new_architecture_list.append(architecture)
        self.architectures = new_architecture_list
        return popped_architecture

    @property
    def seeds(self):
        seeds = {}
        for seed in self._seeds:
            seeds[seed.name] = seed
        return seeds

    def get_seed_by_name(self, seed_name):
        for seed in self._seeds:
            if seed.name == seed_name:
                return seed
        raise Exception("cannot find the seed: ", seed_name)

    def initiate_seed(self):
        for seed_name in self._mutators:
            self._seeds.append(self.MutatorSeed(name=seed_name))

    def initiate_mcmc(self):
        # Hyperparameter Tuning: the range is: [0.313, 0.598] 0.4
        self.mcmc = MCMC(seed_names=list(self.seeds.keys()), p=0.4)

    def increase_seed_reward(self, seed_name):
        seed = self.get_seed_by_name(seed_name)
        seed.increase_reward()

    def choose_seed(self):
        for seed in self._seeds:
            self.mcmc.set_seed_score(seed_name=seed.name, score=seed.score)
        selected_seed_name = self.mcmc.choose_seed(last_used_seed=self.last_used_seed, selection_mode=self.selection_mode)
        selected_seed = self.get_seed_by_name(selected_seed_name)
        selected_architecture = self.pick_architecture()
        architecture_config = selected_architecture.choose_config()
        self.last_used_seed = selected_seed_name
        return selected_architecture.name, architecture_config, selected_seed.name

    def pick_architecture(self, ):
        return np.random.choice(self.architectures)

    def pop_one_architecture(self, ):
        np.random.shuffle(self.architectures)
        self.architectures.pop()

    def architecture_pool_is_full(self):
        if len(self.architectures) <= self.architecture_pool:
            return False
        else:
            return True

    def add_architecture(self, new_architecture_name, new_architecture_config, old_seed_name):
        new_architecture = Architecture(name=new_architecture_name, config=new_architecture_config)
        self.architectures.append(new_architecture)
        while self.architecture_pool_is_full():
            self.pop_one_architecture()

    def increase_reward(self, seed_name, score=1):
        # increase the reward of mutation operator
        selected_seed = self.get_seed_by_name(seed_name=seed_name)
        selected_seed.increase_reward_by_value(score)

    @staticmethod
    def check_inconsistency(self, inconsistency):
        if 10 <= inconsistency:  # 10 is the pre-defined threshold
            main_logger.info(f"New inconsistency issue found!!!")


class MCMC:
    class Seed:
        def __init__(self, name, selected=0, reward_sum=0, epsilon=1e-7):
            self.name = name
            self.score = 0

        @deprecated
        def increase_reward(self, ):
            self.reward_sum += 1

        @deprecated
        def increase_reward_by_value(self, value):
            self.reward_sum += value

        @deprecated
        def increase_selected(self, ):
            self.selected += 1

        def set_score(self, score):
            self.score = score

        # @deprecated
        # def score(self, epsilon=1e-7):
        #     # fitness function will be used to determine the score
        #     return self.reward_sum / (self.selected + 1)

    def __init__(self, p, seed_names=None, seeds=None):
        # self.p = 1 / len(seed_names)
        self.p = p
        self._seeds = [self.Seed(name=seed_name) for seed_name in seed_names]

    @property
    def seeds(self):
        seeds = {}
        for seed in self._seeds:
            seeds[seed.name] = seed
        return seeds

    def index(self, item_name, items):
        for i, item in enumerate(items):
            if item.name == item_name:
                return i
        raise ValueError(f"Fail to index {item_name}")

    def _choose_seed(self, last_used_seed=None):
        self.sort_seeds()  # sort self._seeds according to score
        log_seed(self._seeds, "Seeds")
        main_logger.info(f"The last used seed is: {last_used_seed}")
        if last_used_seed is None:
            # which means it is the first mutation
            selected_seed = self._seeds[np.random.randint(0, len(self._seeds))]
            return selected_seed  # if it is the first time, we just randomly select.
        else:
            k1 = self.index(last_used_seed, self._seeds)
            k2 = -1
            prob = 0
            while np.random.rand() >= prob:
                k2 = np.random.randint(0, len(self._seeds))
                prob = (1 - self.p) ** (k2 - k1)
            selected_seed = self._seeds[k2]
            return selected_seed

    def sort_seeds(self, ):
        # sort seed by its score
        import random
        random.shuffle(self._seeds)
        self._seeds.sort(key=lambda seed: seed.score, reverse=True)

    def choose_seed(self, last_used_seed=None, selection_mode=None):
        # return: seed_name
        if selection_mode == "mcmc":
            selected_seed = self._choose_seed(last_used_seed)
            return selected_seed.name
        elif selection_mode == "random":
            # randomly choose a seed
            selected_seed = self._seeds[np.random.randint(0, len(self._seeds))]
            return selected_seed.name

    def get_seed_by_name(self, seed_name):
        for seed in self._seeds:
            if seed.name == seed_name:
                return seed
        raise Exception("cannot find the seed: ", seed_name)

    def set_seed_score(self, seed_name, score):
        seed = self.get_seed_by_name(seed_name)
        seed.set_score(score)

    @deprecated
    def increase_seed_reward(self, seed_name):
        seed = self.get_seed_by_name(seed_name)
        seed.increase_reward()

    @deprecated
    def increase_seed_reward_by_value(self, seed_name, value):
        seed = self.get_seed_by_name(seed_name)
        seed.increase_reward_by_value(value)
