"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import csv
import pytest
try:
    from io import StringIO
except ImportError:
    from stringio import StringIO

from keras_retinanet.preprocessing import csv_generator

def csv_str(string):
    return csv.reader(StringIO(string))

def test_read_classes():
    assert csv_generator._read_classes(csv_str('')) == {}
    assert csv_generator._read_classes(csv_str('a,1')) == {'a': 1}
    assert csv_generator._read_classes(csv_str('a,1\nb,2')) == {'a': 1, 'b': 2}

def test_read_classes_wrong_format():
    with pytest.raises(ValueError):
        try:
            csv_generator._read_classes(csv_str('a,b,c'))
        except ValueError as e:
            assert str(e).startswith('line 0: format should be')
            raise
    with pytest.raises(ValueError):
        try:
            csv_generator._read_classes(csv_str('a,1\nb,c,d'))
        except ValueError as e:
            assert str(e).startswith('line 1: format should be')
            raise

def test_read_classes_malformed_class_id():
    with pytest.raises(ValueError):
        try:
            csv_generator._read_classes(csv_str('a,b'))
        except ValueError as e:
            assert str(e).startswith("line 0: malformed class ID:")
            raise

    with pytest.raises(ValueError):
        try:
            csv_generator._read_classes(csv_str('a,1\nb,c'))
        except ValueError as e:
            assert str(e).startswith('line 1: malformed class ID:')
            raise

def test_read_classes_duplicate_name():
    with pytest.raises(ValueError):
        try:
            csv_generator._read_classes(csv_str('a,1\nb,2\na,3'))
        except ValueError as e:
            assert str(e).startswith('line 2: duplicate class name')
            raise
