from helpers.config import dataset_list, method_list, dim_keys, cnn_layers


class Cache:
    embedding_cache = {x: {y: None for y in method_list} for x in dataset_list}
    last_embeddings = {x: {y: None for y in method_list} for x in dataset_list}
    model_cache = {x: {y: None for y in method_list} for x in dataset_list}

    @staticmethod
    def load_embedding_cache(dataset, method, value):
        Cache.embedding_cache[dataset][method] = value

    @staticmethod
    def get_embedding_cache(dataset, method):
        return Cache.embedding_cache[dataset][method]

    @staticmethod
    def load_last_embeddings(dataset, method, value):
        Cache.last_embeddings[dataset][method] = value

    @staticmethod
    def get_last_embeddings(dataset, method):
        return Cache.last_embeddings[dataset][method]

    @staticmethod
    def load_model_cache(dataset, method, value):
        Cache.model_cache[dataset][method] = value

    @staticmethod
    def get_model_cache(dataset, method):
        return Cache.model_cache[dataset][method]


class TriMapInversionCache:
    embedding_cache = {x: {str(y): None for y in dim_keys['TriMap']} for x in dataset_list}
    model_cache = {x: {str(y): None for y in dim_keys['TriMap']} for x in dataset_list}

    @staticmethod
    def load_embedding_cache(dataset, key, value):
        TriMapInversionCache.embedding_cache[dataset][key] = value

    @staticmethod
    def get_embedding_cache(dataset, key):
        return TriMapInversionCache.embedding_cache[dataset][key]

    @staticmethod
    def load_model_cache(dataset, key, value):
        TriMapInversionCache.model_cache[dataset][key] = value

    @staticmethod
    def get_model_cache(dataset, key):
        return TriMapInversionCache.model_cache[dataset][key]


class CNNLayerInversionCache:
    embedding_cache = {x: {str(y): None for y in cnn_layers} for x in dataset_list}
    model_cache = {x: {str(y): None for y in cnn_layers} for x in dataset_list}

    @staticmethod
    def load_embedding_cache(dataset, key, value):
        CNNLayerInversionCache.embedding_cache[dataset][key] = value

    @staticmethod
    def get_embedding_cache(dataset, key):
        return CNNLayerInversionCache.embedding_cache[dataset][key]

    @staticmethod
    def load_model_cache(dataset, key, value):
        CNNLayerInversionCache.model_cache[dataset][key] = value

    @staticmethod
    def get_model_cache(dataset, key):
        return CNNLayerInversionCache.model_cache[dataset][key]
