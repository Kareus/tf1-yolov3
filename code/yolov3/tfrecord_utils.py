import tensorflow as tf
from itertools import tee

class TFRecordIterator:
    def __init__(self, path, compression=None):
        self._core = tf.python_io.tf_record_iterator(path, tf.python_io.TFRecordOptions(compression))
        self._iterator = iter(self._core)
        self._iterator, self._iterator_temp = tee(self._iterator)
        self._total_cnt = sum(1 for _ in self._iterator_temp)

    def _read_value(self, feature):
        if len(feature.int64_list.value) > 0:
            return feature.int64_list.value

        if len(feature.bytes_list.value) > 0:
            return feature.bytes_list.value

        if len(feature.float_list.value) > 0:
            return feature.float_list.value

        return None

    def _read_features(self, features):
        d = dict()
        for data in features:
            d[data] = self._read_value(features[data])
        return d

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        record = next(self._iterator)
        example = tf.train.Example()
        example.ParseFromString(record)
        return self._read_features(example.features.feature)

    def count(self):
        return self._total_cnt

    