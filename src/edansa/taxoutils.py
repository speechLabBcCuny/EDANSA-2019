from typing import Dict, Union, Optional, Type
from collections import Counter
from collections.abc import MutableMapping


def find_upper_taxo(taxo):

    if '.' in taxo:
        taxo_a = taxo.split('.')
    else:
        taxo_a = taxo[:]

    if set(taxo_a) == set('X'):
        return '.'.join(taxo_a)
    if 'X' in taxo_a:
        taxo_a = [x if x != 'X' else '0' for x in taxo_a]
    # -1 because we do not change first bit
    for i in range(len(taxo_a) - 1):
        if taxo_a[-(i + 1)] == '0':
            continue
        else:
            taxo_a[-(i + 1)] = '0'
            break

    taxo_a = '.'.join(taxo_a)
    return taxo_a


def get_root_taxos(org_taxo):
    root_taxos = []
    upper_taxo = find_upper_taxo(org_taxo)
    previous_taxo = org_taxo
    while upper_taxo != previous_taxo:
        root_taxos.append(upper_taxo)
        previous_taxo = upper_taxo
        upper_taxo = find_upper_taxo(previous_taxo)
    return root_taxos


assert ['1.1.0', '1.0.0'] == get_root_taxos('1.1.1')


def row2yaml_codev1(row, excell_names2code):
    if row['Specific Category'] in ['Songb', 'SongB']:
        row['Specific Category'] = 'Songbird'

    if row['Category'] in ['Mamm']:
        row['Category'] = 'Mam'
    # 'S4A10288_20190729_033000_unknown.wav', # 'Anthro/Bio': 'Uknown', no other data

    if row['Anthro/Bio'] in ['Uknown', 'Unknown']:
        row['Anthro/Bio'] = ''

    code = [row['Anthro/Bio'], row['Category'], row['Specific Category']]

    # place X for unknown topology
    # '0' is reserved for 'other'
    code = [i if i != '' else 'X' for i in code]

    # place X for unknown topology
    # '0' is reserved for 'other'
    # print(code)
    code = [i if i != '' else 'X' for i in code]
    for c in code:
        if '/' in c:
            raise NotImplementedError(
                f"row has wrong info about categories, '/' found: {row}")

    if code == ['X', 'X', 'X']:
        yaml_code = 'X.X.X'
    elif code[2] != 'X':
        yaml_code = excell_names2code[code[2].lower()]
    elif code[1] != 'X':
        yaml_code = excell_names2code[code[1].lower()]
    elif code[0] != 'X':
        yaml_code = excell_names2code[code[0].lower()]
    else:
        print(code)
        raise ValueError(f'row does not belong to any toplogy: {row}')

    return yaml_code


def row2yaml_codev2(row, excell_names2code):
    '''
        
    
    '''
    # we are hard coding this because this function for a specific template
    #
    yaml_codes = []
    excell_class_names = {
        'whim', 'amgp', 'shorebird', 'spsa', 'paja', 'hola', 'sngo', 'lalo',
        'savs', 'wipt', 'nora', 'atsp', 'wisn', 'wfgo', 'sesa', 'glgu', 'bird',
        'mam', 'loon', 'helo', 'auto', 'jet', 'corv', 'dgs', 'flare', 'bio',
        'bear', 'crane', 'fly', 'airc', 'hum', 'truck', 'rain', 'bug', 'mosq',
        'geo', 'mach', 'woof', 'deer', 'water', 'anth', 'songb', 'woop', 'car',
        'rapt', 'sil', 'seab', 'wind', 'owl', 'meow', 'mous', 'prop', 'grous',
        'weas', 'hare', 'shrew'
    }
    row_lower = {k.lower(): v for k, v in row.items()}
    for header in row_lower:
        if header not in excell_class_names:
            print(f'WARNING: header {header} is not accepted as class name')

    for excell_class_name in excell_class_names:
        # value is 1 or 0
        class_exists_or_not = row_lower.get(excell_class_name, None)
        # print(class_exists_or_not)
        class_exists_or_not = str(class_exists_or_not)
        if class_exists_or_not == '1':
            yaml_code = excell_names2code[excell_class_name]
            yaml_codes.append(yaml_code)

    if len(yaml_codes) != len(set(yaml_codes)):
        print(row)
        raise Exception(
            f'input excell have non-unique class names, yaml code counts: {Counter(yaml_codes)}'
        )

    yaml_codes.sort()
    return yaml_codes


def megan_excell_row2yaml_code(row: Dict,
                               excell_names2code: Dict = None,
                               version='V2'):
    '''Megan style labels to nna yaml topology V1.

    Row is a mapping, with 3 topology levels, function starts from most specific
    category and goes to most general one, when a mapping is found, returns
    corresponding code such as 0.2.0 for plane.

    Args:
        row = dictinary with following keys
                'Anthro/Bio','Category','Specific Category'
        excell_names2code = mapping from names to topology code
        version = Version of the function, 
            v1: single taxonomy per sample
            v2: multi taxonomy per sample, returns a list

    '''
    if excell_names2code is None:
        excell_names2code = {
            'SNGO': '4.1.10.0',
            'LALO': '4.1.10.1',
            'SAVS': '4.1.10.2',
            'WIPT': '4.1.10.3',
            'NORA': '4.1.10.4',
            'ATSP': '4.1.10.5',
            'WISN': '4.1.10.6',
            'WFGO': '4.1.10.7',
            'SESA': '4.1.10.8',
            'GLGU': '4.1.10.9',
            'anth': '0.0.0',
            'auto': '0.1.0',
            'car': '0.1.2',
            'truck': '0.1.1',
            'prop': '0.2.1',
            'helo': '0.2.2',
            'jet': '0.2.3',
            'mach': '0.3.0',
            'bio': '1.0.0',
            'bird': '1.1.0',
            'crane': '1.1.11',
            'corv': '1.1.12',
            'hum': '1.1.1',
            'shorb': '1.1.2',
            'rapt': '1.1.4',
            'owl': '1.1.6',
            'woop': '1.1.9',
            'bug': '1.3.0',
            'dgs': '1.1.7',
            'flare': '0.4.0',
            'fox': '1.2.4',
            'geo': '2.0.0',
            'grouse': '1.1.8',
            'grous': '1.1.8',
            'loon': '1.1.3',
            'mam': '1.2.0',
            'bear': '1.2.2',
            'plane': '0.2.0',
            'ptarm': '1.1.8',
            'rain': '2.1.0',
            'seab': '1.1.5',
            'mous': '1.2.1',
            'deer': '1.2.3',
            'woof': '1.2.4',
            'weas': '1.2.5',
            'meow': '1.2.6',
            'hare': '1.2.7',
            'shrew': '1.2.8',
            'fly': '1.3.2',
            'silence': '3.0.0',
            'sil': '3.0.0',
            'songbird': '1.1.10',
            'songb': '1.1.10',
            'unknown': 'X.X.X',
            'water': '2.2.0',
            'x': 'X.X.X',
            'airc': '0.2.0',
            'mosq': '1.3.1',
            'wind': '2.3.0',
        }

    if version == 'V1':
        yaml_code = row2yaml_codev1(row, excell_names2code)
    elif version == 'V2':
        # returns a list
        yaml_code = row2yaml_codev2(row, excell_names2code)
    else:
        raise ValueError(
            f'This version is not implemented at megan_excell_row2yaml_code {version}'
        )

    return yaml_code


# taxonomy YAML have an issue that leafes has a different structure then previous
# orders, I should change that.
class Taxonomy(MutableMapping):
    """A dictionary that holds taxonomy structure.

    transforms edge keys from x.y.z to just last bit z 
    
    """

    def __init__(self, *args, **kwargs):
        self._init_end = False
        self._store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys
        self._edges = self.flatten(self._store)
        self.shorten_edge_keys(self._store)
        self._init_end = True

    @property
    def edges(self,):
        """Property of _edges."""
        return self._edges

    @edges.setter
    def edges(self, value):
        if self._init_end:
            raise NotImplementedError('Edges and taxonomy are Immutable')
        else:
            self._edges = value

    @edges.getter
    def edges(self,):
        return self._edges

    def __getitem__(self, key):
        key = self._keytransform(key)
        if isinstance(key, list):
            data = self._store
            for k in key:
                data = data[k]
            return data
        return self._store[key]

        # trying to implement general access by single key or multiple with dot
        # current_order = self._store[key[0]]
        # if len(key)==1:
        #     return current_order
        # keys = self._store[self._keytransform(key)]
        # for k in keys[:-1]:
        #     current_order = current_order[k]

        # return current_order[key]

    def __setitem__(self, key, value):
        if self._init_end:
            raise NotImplementedError('You cannot update after initilization.')
        else:
            self._store[key] = value

    def __delitem__(self, key):
        if self._init_end:
            raise NotImplementedError('You cannot update after initilization.')
        else:
            del self._store[key]
        # del self._store[self._keytransform(key)]
        # self.edges = self.flatten(self._store)

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def _keytransform(self, key):
        if isinstance(key, str):
            return key.split('.')
        elif isinstance(key, list):
            return key
        return key

    def flatten(self, d):
        out = {}
        for key, val in d.items():
            if isinstance(val, dict):
                deeper = self.flatten(val).items()
                out.update(deeper)
            else:
                out[key] = val
        return out

    def shorten_edge_keys(self, d):
        for key, val in list(d.items()):
            del d[key]
            if isinstance(val, dict):
                d[key.split('.')[-1]] = self.shorten_edge_keys(val)
            else:
                d[key.split('.')[-1]] = val
        return d


excell_all_headers = [
    'data_version',
    'Annotator',
    'region',
    'Site ID',
    'Comments',
    'File Name',
    'Date',
    'Start Time',
    'End Time',
    'Length',
    'WHIM',
    'AMGP',
    'SHOREBIRD',
    'SPSA',
    'PAJA',
    'HOLA',
    'SNGO',
    'LALO',
    'SAVS',
    'WIPT',
    'NORA',
    'ATSP',
    'WISN',
    'WFGO',
    'SESA',
    'GLGU',  # new from John's labels
    'Anth',
    'Bio',
    'Geo',
    'Sil',
    'Auto',
    'Airc',
    'Mach',
    'Flare',
    'Bird',
    'Mam',
    'Bug',
    'Wind',
    'Rain',
    'Water',
    'Truck',
    'Car',
    'Prop',
    'Helo',
    'Jet',
    'Corv',
    'SongB',
    'DGS',
    'Grous',
    'Crane',
    'Loon',
    'SeaB',
    'Owl',
    'Hum',
    'Rapt',
    'Woop',
    'ShorB',
    'Woof',
    'Bear',
    'Mous',
    'Deer',
    'Weas',
    'Meow',
    'Hare',
    'Shrew',
    'Mosq',
    'Fly',
    'Clip Path',
    'Reviewed',
    'extra_tags'
]
excell_label_headers = [
    'WHIM',
    'AMGP',
    'SHOREBIRD',
    'SPSA',
    'PAJA',
    'HOLA',
    'SNGO',
    'LALO',
    'SAVS',
    'WIPT',
    'NORA',
    'ATSP',
    'WISN',
    'WFGO',
    'SESA',
    'GLGU',  # new from John's labels
    'Anth',
    'Bio',
    'Geo',
    'Sil',
    'Auto',
    'Airc',
    'Mach',
    'Flare',
    'Bird',
    'Mam',
    'Bug',
    'Wind',
    'Rain',
    'Water',
    'Truck',
    'Car',
    'Prop',
    'Helo',
    'Jet',
    'Corv',
    'SongB',
    'DGS',
    'Grous',
    'Crane',
    'Loon',
    'SeaB',
    'Owl',
    'Hum',
    'Rapt',
    'Woop',
    'ShorB',
    'Woof',
    'Bear',
    'Mous',
    'Deer',
    'Weas',
    'Meow',
    'Hare',
    'Shrew',
    'Mosq',
    'Fly'
]

excell_names2code = {
    'Anth': '0.0.0',
    'Bio': '1.0.0',
    'Geo': '2.0.0',
    'Sil': '3.0.0',
    'Auto': '0.1.0',
    'Airc': '0.2.0',
    'Mach': '0.3.0',
    'Flare': '0.4.0',
    'Bird': '1.1.0',
    'sngo': '4.1.10.0',
    'lalo': '4.1.10.1',
    'savs': '4.1.10.2',
    'wipt': '4.1.10.3',
    'nora': '4.1.10.4',
    'atsp': '4.1.10.5',
    'wisn': '4.1.10.6',
    'wfgo': '4.1.10.7',
    'sesa': '4.1.10.8',
    'glgu': '4.1.10.9',
    'whim': '4.1.10.10',
    'amgp': '4.1.10.11',
    'shorebird': '4.1.10.12',
    'spsa': '4.1.10.13',
    'paja': '4.1.10.14',
    'hola': '4.1.10.15',
    'LALO': '4.1.10.1',
    'SAVS': '4.1.10.2',
    'WIPT': '4.1.10.3',
    'NORA': '4.1.10.4',
    'SNGO': '4.1.10.0',
    'ATSP': '4.1.10.5',
    'WISN': '4.1.10.6',
    'WFGO': '4.1.10.7',
    'SESA': '4.1.10.8',
    'GLGU': '4.1.10.9',
    'WHIM': '4.1.10.10',
    'AMGP': '4.1.10.11',
    'SHOREBIRD': '4.1.10.12',
    'SPSA': '4.1.10.13',
    'PAJA': '4.1.10.14',
    'HOLA': '4.1.10.15',
    'Mam': '1.2.0',
    'Bug': '1.3.0',
    'Wind': '2.3.0',
    'Rain': '2.1.0',
    'Water': '2.2.0',
    'Truck': '0.1.1',
    'Car': '0.1.2',
    'Prop': '0.2.1',
    'Helo': '0.2.2',
    'Jet': '0.2.3',
    'Corv': '1.1.12',
    'SongB': '1.1.10',
    'DGS': '1.1.7',
    'Grous': '1.1.8',
    'Crane': '1.1.11',
    'Loon': '1.1.3',
    'SeaB': '1.1.5',
    'Owl': '1.1.6',
    'Hum': '1.1.1',
    'Rapt': '1.1.4',
    'Woop': '1.1.9',
    'ShorB': '1.1.2',
    'Woof': '1.2.4',
    'Bear': '1.2.2',
    'Mous': '1.2.1',
    'Deer': '1.2.3',
    'Weas': '1.2.5',
    'Meow': '1.2.6',
    'Hare': '1.2.7',
    'Shrew': '1.2.8',
    'Mosq': '1.3.1',
    'Fly': '1.3.2',
    'anth': '0.0.0',
    'auto': '0.1.0',
    'car': '0.1.2',
    'truck': '0.1.1',
    'prop': '0.2.1',
    'helo': '0.2.2',
    'jet': '0.2.3',
    'mach': '0.3.0',
    'bio': '1.0.0',
    'bird': '1.1.0',
    'crane': '1.1.11',
    'corv': '1.1.12',
    'hum': '1.1.1',
    'shorb': '1.1.2',
    'rapt': '1.1.4',
    'owl': '1.1.6',
    'woop': '1.1.9',
    'bug': '1.3.0',
    'bugs': '1.3.0',
    'insect': '1.3.0',
    'dgs': '1.1.7',
    'flare': '0.4.0',
    'fox': '1.2.4',
    'geo': '2.0.0',
    'grouse': '1.1.8',
    'grous': '1.1.8',
    'loon': '1.1.3',
    'loons': '1.1.3',
    'mam': '1.2.0',
    'bear': '1.2.2',
    'plane': '0.2.0',
    'ptarm': '1.1.8',
    'rain': '2.1.0',
    'seab': '1.1.5',
    'seabirds': '1.1.5',
    'mous': '1.2.1',
    'deer': '1.2.3',
    'woof': '1.2.4',
    'weas': '1.2.5',
    'meow': '1.2.6',
    'hare': '1.2.7',
    'shrew': '1.2.8',
    'fly': '1.3.2',
    'silence': '3.0.0',
    'sil': '3.0.0',
    'songbird': '1.1.10',
    'songb': '1.1.10',
    'birdsong': '1.1.10',
    'unknown': 'X.X.X',
    'water': '2.2.0',
    'x': 'X.X.X',
    'airc': '0.2.0',
    'aircraft': '0.2.0',
    'mosq': '1.3.1',
    'wind': '2.3.0',
    'windy': '2.3.0',
}

taxo_code2excell_names = {
    '4.1.10.0': 'SNGO',
    '4.1.10.1': 'LALO',
    '4.1.10.2': 'SAVS',
    '4.1.10.3': 'WIPT',
    '4.1.10.4': 'NORA',
    '4.1.10.5': 'ATSP',
    '4.1.10.6': 'WISN',
    '4.1.10.7': 'WFGO',
    '4.1.10.8': 'SESA',
    '4.1.10.9': 'GLGU',
    '0.0.0': 'Anth',
    '1.0.0': 'Bio',
    '2.0.0': 'Geo',
    '3.0.0': 'Sil',
    '0.1.0': 'Auto',
    '0.1.2': 'Car',
    '0.2.0': 'Airc',
    '0.3.0': 'Mach',
    '0.4.0': 'Flare',
    '1.1.0': 'Bird',
    '1.2.0': 'Mam',
    '1.3.0': 'Bug',
    '2.3.0': 'Wind',
    '2.1.0': 'Rain',
    '2.2.0': 'Water',
    '0.1.1': 'Truck',
    '0.2.1': 'Prop',
    '0.2.2': 'Helo',
    '0.2.3': 'Jet',
    '1.1.12': 'Corv',
    '1.1.10': 'SongB',
    '1.1.7': 'DGS',
    '1.1.8': 'Grous',
    '1.1.11': 'Crane',
    '1.1.3': 'Loon',
    '1.1.5': 'SeaB',
    '1.1.6': 'Owl',
    '1.1.1': 'Hum',
    '1.1.4': 'Rapt',
    '1.1.9': 'Woop',
    '1.1.2': 'ShorB',
    '1.2.4': 'Woof',
    '1.2.2': 'Bear',
    '1.2.1': 'Mous',
    '1.2.3': 'Deer',
    '1.2.5': 'Weas',
    '1.2.6': 'Meow',
    '1.2.7': 'Hare',
    '1.2.8': 'Shrew',
    '1.3.1': 'Mosq',
    '1.3.2': 'Fly'
}
