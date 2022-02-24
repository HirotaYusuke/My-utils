from typing import Optional
import pandas as pd
from .base import Feature
from ..utils import timer
from pathlib import Path
import os


Feature.dir = 'features'

def save(train: pd.DataFrame, test: Optional[pd.DataFrame] = None):
    for name in train.columns:
        f_cls = Feature_class()
        f_cls.name = name
        f_cls.train_path = Path(f_cls.dir) / f'{name}_train.ftr'
        f_cls.test_path = Path(f_cls.dir) / f'{name}_test.ftr'
        if not name in test.columns:
            f_cls.train_only = True
        f_cls.run(train, test).save()


class Feature_class(Feature):
    def create_features(self, train: pd.Series, test: Optional[pd.Series] = None):
        self.train[f'{self.name}'] = train[f'{self.name}']
        if not self.train_only:
            self.test[f'{self.name}'] = test[f'{self.name}']

    def run(self, train_row: pd.Series, test_row: Optional[pd.Series] = None):
        with timer(self.name):
            self.create_features(train_row, test_row)
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self