'''Module handling importing data from external sources.

'''
from argparse import ArgumentError
from typing import Dict, Union, Optional, Type, List

from pathlib import Path
from collections.abc import MutableMapping
from collections import Counter

import numpy as np

from datetime import datetime
import csv

import edansa.taxoutils
import edansa.utils
import edansa.clippingutils


class Audio():
    """Single audio sample within the dataset.
    """

    def __init__(
        self,
        path: Union[str, Path],
        length_seconds: float,
        taxo_codes: List[str] = None,
        clipping=None,  # Optional[np.array] 
        # shape = (# of 10 seconds,number of channels)
        location_id=None,
        region=None,
        data_version=None,
        # annotator=None,
        comments=None,
        original_recording_path=None,
        timestamps=None,
    ):
        self.path = Path(path)
        self.name = self.path.name
        self.suffix = self.path.suffix
        self.clipping = clipping
        self.data = np.empty(0)  # suppose to be np.array
        self.sr: Optional[int] = None  # sampling rate
        self.location_id = location_id  #
        self.region = region
        self.samples = []
        self.data_version = data_version
        # self.annotator = annotator
        self.comments = comments
        self.original_recording_path = original_recording_path
        self.timestamps = timestamps  # (start,end)
        self.length = length_seconds  # in seconds
        self.taxo_codes = taxo_codes

    def __str__(self,):
        return str(self.name)

    def __repr__(self,):
        return f'{self.path}, length:{self.length}'

    def pick_channel_by_clipping(self, excerpt_length):
        if len(self.data.shape) == 1:
            return None
        if self.clipping is None:
            raise ValueError(f'{self.path} does not have clipping information.')

        cleaner_channel_indexes = np.argmin(self.clipping, axis=1)
        new_data = np.empty(self.data.shape[-1], dtype=self.data.dtype)

        excpert_len_jump = self.sr * excerpt_length

        for ch_i, data_i in zip(cleaner_channel_indexes,
                                range(0, self.data.shape[-1],
                                      excpert_len_jump)):
            new_data[data_i:data_i +
                     excpert_len_jump] = self.data[ch_i, data_i:data_i +
                                                   excpert_len_jump]

        self.data = new_data[:]

        return 1

    def load_data(
        self,
        dtype=np.int16,
        store=True,
        resample_rate=None,
    ):
        sound_array, sr = edansa.clippingutils.load_audio(
            self.path,
            dtype=dtype,
            backend='pydub',
            resample_rate=resample_rate)
        if store:
            self.data = sound_array
            self.sr = sr
        else:
            return sound_array, sr

        # self.data_version = data_version
        # self.annotator = annotator
        # self.comments = comments
        # self.original_recording_path = original_recording_path
        # self.timestamps = timestamps  # (start,end)
        # self.length = length_seconds  # in seconds
        # self.taxo_codes = taxo_codes
    def get_data_by_value(self,):
        # print(self.path)
        if (self.data.size is None) or (self.data.size == 0):
            return self.load_data(store=False)
        else:
            return self.data.copy(), self.sr

    def load_info(self,
                  row,
                  excell_names2code=None,
                  version='V2',
                  dataset_folder=None):

        fname = row['Clip Path']
        length = row['Length']
        times_str_start = row['Date'] + ' ' + row['Start Time']
        times_str_end = row['Date'] + ' ' + row['End Time']
        times_frmt = '%m/%d/%Y %H:%M:%S.%f'
        start_time = datetime.strptime(times_str_start, times_frmt)
        end_time = datetime.strptime(times_str_end, times_frmt)

        length_time = datetime.strptime(length, '%H:%M:%S.%f')
        total_seconds = (length_time - length_time.replace(
            hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        if dataset_folder is not None:
            fname = Path(dataset_folder) / fname
        self.path = Path(fname)
        self.name = self.path.name
        self.suffix = self.path.suffix
        self.length = float(total_seconds)
        site_id = row['Site ID'].strip().lower()
        region = row['region'].strip().lower()
        self.location_id = site_id
        self.region = region
        self.data_version = row['batch']
        # self.annotator = row['Annotator']
        # self.comments = row['Comments']
        # self.original_recording_path = row['File Name']
        self.timestamps = (start_time, end_time)

        if excell_names2code is not None:
            taxonomy_codes = edansa.taxoutils.megan_excell_row2yaml_code(
                row, excell_names2code, version=version)
            self.taxo_codes = taxonomy_codes

    def sample_count(
        self,
        excerpt_length=10,
        sample_length_limit=2,
    ):
        '''count number of samples can be extracted from the audio file
        '''
        if self.length == -1 or self.length is None:
            return -1

        if self.length >= sample_length_limit:
            sample_count = self.length // excerpt_length
            left_overs = (self.length % excerpt_length)
            if left_overs > sample_length_limit and left_overs > 0:
                sample_count += 1
        else:
            sample_count = 0

        return sample_count

    def data_to_samples(self, excerpt_len=10):
        '''
            self.data is a numpy array of shape (channels,n_samples)
            excerpt_len is length of excerpt in seconds
        '''

        assert len(self.data.shape) == 2
        # print(self.data.shape)
        # print(excerpt_len ,self.sr)
        excerpt_sample_size = excerpt_len * self.sr

        data_len_sec = self.length
        # if data is smaller than the expected size, then we need to
        # just padd it with zeros
        if data_len_sec < excerpt_len:
            missing_element_count = int(excerpt_sample_size -
                                        self.data.shape[1])
            padded_data = np.pad(self.data,
                                 ((0, 0), (0, missing_element_count)),
                                 'constant',
                                 constant_values=(0, 0))
            self.samples = [padded_data]

        # if it is exactly the expected size, then we can just return it
        elif data_len_sec == excerpt_len:
            self.samples = [self.data]
        # if it is bigger, then we need to divide it into chunks
        else:
            sample_list = self.divide_long_sample(
                data_len_sec,
                excerpt_len,
            )

            self.samples = sample_list

    def divide_long_sample(
        self,
        data_len_sec,
        excerpt_len,
    ):
        '''
            divide the long audio file into chunks of size excerpt_len

            if sample shorter than excerpt_len, then it pads rest with 0s
            if sample longer than excerpt_len, then it cuts it into pieces
                however, if one of the pieces is shorter than 5 (!) seconds, 
                then it pads it with zeros otherwise ignores it.

            data_len_sec: length of the audio file in seconds
            excerpt_len: length of the samples to be extracted in seconds

        '''
        excerpt_sample_size = excerpt_len * self.sr

        excerpt_count = data_len_sec // excerpt_len
        # first process part of the sample that is at least excerpt_len long.
        data_trim_point = int(excerpt_count * excerpt_len * self.sr)
        samples = self.data[:, :data_trim_point].reshape(
            self.data.shape[0], -1, int(excerpt_sample_size))
        sample_list = []
        for sample_i in range(samples.shape[1]):
            sample_list.append(samples[:, sample_i, :])
        # if left over is >= 5 seconds, pad with zeros to make a new sample
        left_over_len = data_len_sec % excerpt_len
        if left_over_len >= 5:
            missing_element_count = int(excerpt_sample_size -
                                        (self.data.shape[1] - data_trim_point))
            padded_data = np.pad(self.data[:, data_trim_point:],
                                 ((0, 0), (0, missing_element_count)),
                                 'constant',
                                 constant_values=(0, 0))
            sample_list.append(padded_data)
        return sample_list

    def get_row_format(self):

        time_style = '%H:%M:%S.%f'
        time_diff = (self.timestamps[1] - self.timestamps[0])
        time_diff_total_seconds = time_diff.total_seconds()
        time_diff = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0) + time_diff

        start_time = datetime.strftime(self.timestamps[0], time_style)
        end_time = datetime.strftime(self.timestamps[1], time_style)
        if self.length != time_diff_total_seconds:
            print('Warning: length != (end-start),\n' +
                  f'{self.length} != {time_diff_total_seconds}')
        length = datetime.strftime(time_diff, time_style)

        date_start = datetime.strftime(self.timestamps[0], '%m/%d/%Y')

        row = {
            'batch': self.data_version,
            # 'Annotator': self.annotator,
            'region': self.region,
            'Site ID': self.location_id,
            # 'Comments': self.comments,
            # 'File Name': self.original_recording_path,
            'Date': date_start,
            'Start Time': start_time,
            'End Time': end_time,
            'Length': length,
        }

        for header in edansa.taxoutils.excell_label_headers:
            if edansa.taxoutils.excell_names2code[header] in self.taxo_codes:
                row[header] = '1'
            else:
                row[header] = '0'

        row['Clip Path'] = str(self.path)

        if None in row.values():
            print('Warning: some values of audio sample is missing for excell')
            print(row)

        return row


class Dataset(MutableMapping):
    """A dictionary that holds data points."""

    def __init__(
        self,
        csv_path_or_rows=None,
        dataset_name_v='',
        excerpt_len=10,
        min_allowed_sample_length=2,
        dataset_cache_folder='',
        data_dict=None,
        excell_names2code=None,
        dataset_folder=None,
    ):
        self.store = dict()
        if data_dict is not None:
            self.update(dict(**data_dict))  # use the free update to set keys
        self.excerpt_length = excerpt_len  # in seconds
        self.min_allowed_sample_length = min_allowed_sample_length
        self.dataset_name_v = dataset_name_v
        self.excell_names2code = excell_names2code
        self.csv_path_or_rows = csv_path_or_rows
        self.dataset_folder = dataset_folder
        if dataset_cache_folder == '':
            self.dataset_cache_folder = ''
        else:
            self.dataset_cache_folder = Path(dataset_cache_folder)
        if csv_path_or_rows is not None:
            self.load_csv(self.csv_path_or_rows,
                          self.dataset_name_v,
                          self.dataset_cache_folder,
                          excell_names2code=self.excell_names2code,
                          dataset_folder=self.dataset_folder)

    def __getitem__(self, key):
        return self.store[self._keytransform(key)]

    def __setitem__(self, key, value):
        self.store[self._keytransform(key)] = value

    def __delitem__(self, key):
        del self.store[self._keytransform(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def _keytransform(self, key):
        return key

    def load_csv(self,
                 csv_path_or_rows,
                 dataset_name_v='',
                 dataset_cache_folder='',
                 excell_names2code=None,
                 dataset_folder=None):
        """read path, len of megan labeled files from csv file, (lnength col.)
        store them in a dataimport.dataset, keys are gonna be sample file path
        """
        if dataset_folder is None:
            dataset_folder = self.dataset_folder
        # if excell_names2code is None and self.excell_names2code is None:
        #     raise ArgumentError('excell_names2code shoudl be provided')
        #src_path = '/scratch/enis/data/nna/labeling/megan/AudioSamplesPerSite/'
        # ffp = src_path + 'meganLabeledFiles_wlenV1.txt'
        if isinstance(csv_path_or_rows, str):
            dataset_rows = self._read_csv_file(csv_path_or_rows)
        else:
            dataset_rows = csv_path_or_rows[:]
        for row in dataset_rows:
            if dataset_folder is not None:
                fname = str(Path(dataset_folder) / row['Clip Path'])
            else:
                fname = row['Clip Path']
            self.store[fname] = Audio('', -1)
            self.store[fname].load_info(row,
                                        excell_names2code=excell_names2code,
                                        dataset_folder=dataset_folder)

        # path has to be inque but is file names are unique ?

        print(len(set([i.name for i in self.store.values()])),
              len(list(self.store.keys())))
        # assert len(set([i.name for i in self.store.values()
        #    ])) == len(list(self.store.keys()))

        print('dataset_cache_folder is ', dataset_cache_folder)
        self.dataset_name_v = dataset_name_v
        self.dataset_cache_folder = dataset_cache_folder

        return True

    def _read_csv_file(self, csv_path):
        with open(csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            reader = list(reader)
            reader_strip = []
            for row in reader:
                row = {r: row[r].strip() for r in row}
                reader_strip.append(row)
            reader = reader_strip.copy()
        return reader

    def export_csv(self, csv_path):
        """export dataset to csv file"""
        rows = []
        for key in self.store:
            rows.append(self.store[key].get_row_format())

        fieldnames = edansa.taxoutils.excell_all_headers
        edansa.utils.write_csv(csv_path, rows, fieldnames=fieldnames)

        return True

    def dataset_clipping_percentage(self, output_folder='') -> tuple:
        """for given dataset_name_version, calculate clipping info

            check if there is already a previous calculation and load that
            ex format "output/megan_1,0.pkl"

        """
        if output_folder == '':
            print('change to dataset_cache_folder', self.dataset_cache_folder)
            output_folder = self.dataset_cache_folder
        if output_folder and self.dataset_name_v:
            dict_output_path = Path(output_folder) / (self.dataset_name_v +
                                                      '_1,0.pkl')
            clipping_error_path = Path(output_folder) / (self.dataset_name_v +
                                                         '_1,0_error.pkl.pkl')
        else:
            print('WENT to else :(')
            print(output_folder, self.dataset_name_v)
            dict_output_path = None
            clipping_error_path = None

        if dict_output_path and dict_output_path.exists():
            clipping_results = np.load(dict_output_path, allow_pickle=True)[()]
            if clipping_error_path and clipping_error_path.exists():
                clipping_errors = np.load(clipping_error_path,
                                          allow_pickle=True)[()]
            else:
                clipping_errors = []
            return clipping_results, clipping_errors
        else:
            msg = ((f'Could not find clipping info at {dict_output_path} ' +
                    'calculating.'))
            print(msg)
            path_list = []
            for key in self.store:
                path_list.append(self.store[key].path)

            all_results_dict, files_w_errors = edansa.clippingutils.run_task_save(
                path_list, self.dataset_name_v, output_folder, 1.0)
            return all_results_dict, files_w_errors

    def update_samples_w_clipping_info(self, output_folder=''):

        all_results_dict, files_w_errors = self.dataset_clipping_percentage(
            output_folder)
        del files_w_errors
        for key in self.store:
            clipping = all_results_dict.get(str(self.store[key].path), None)
            if clipping is not None:
                self.store[key].clipping = clipping

    def load_audio_files(
        self,
        cached_dict_path=None,
        dtype=np.int16,
    ):
        if cached_dict_path is not None and cached_dict_path:
            print("loading from cache at {}".format(cached_dict_path))
            cached_dict = np.load(cached_dict_path, allow_pickle=True)[()]
        else:
            print('no cache found, loading original files')
            cached_dict = {}
        for key, value in self.store.items():
            data = cached_dict.get(str(value.path), None)
            if data is None:
                sound_array, sr = nna.clippingutils.load_audio(value.path,
                                                               dtype=dtype,
                                                               backend='pydub')
            else:
                sound_array, sr = data

            self.store[key].data = sound_array
            self.store[key].sr = sr

    def pick_channel_by_clipping(self):
        for _, audio_sample in self.store.items():
            if audio_sample.clipping is None:
                print('Found audio samples with no clipping info, loading')
                self.update_samples_w_clipping_info()

        for _, audio_sample in self.store.items():
            # this is a function of Audio
            audio_sample.pick_channel_by_clipping(self.excerpt_length)

    def create_cache_pkl(self, output_file_path):
        '''save data files of samples as pkl.
        '''
        data_dict = {}
        if Path(output_file_path).exists():
            raise ValueError(f'{output_file_path} already exists')
        for value in self.store.values():
            data_dict[str(value.path)] = value.data, value.sr

        with open(output_file_path, 'wb') as f:
            np.save(f, data_dict)

    def count_samples_per_taxo_code(self,
                                    sample_length_limit=None,
                                    version='V2'):
        """Go through rows of the excell and count category population

            returns:
                taxo_code_counter: {'0.0.1':12,...}
        """
        no_taxo_code = []
        taxo_code_counter = Counter()
        if sample_length_limit is None:
            sample_length_limit = self.min_allowed_sample_length

        for audio_ins in self.store.values():
            if audio_ins.taxo_codes is None:
                no_taxo_code.append(audio_ins)
                continue
            sample_count = audio_ins.sample_count(
                excerpt_length=self.excerpt_length,
                sample_length_limit=sample_length_limit)

            if version == 'V1':
                taxo_code_counter.update({audio_ins.taxo_codes: sample_count})
            elif version == 'V2':
                taxo_code_counter.update({
                    taxo_code: sample_count
                    for taxo_code in audio_ins.taxo_codes
                })
        if no_taxo_code:
            msg = 'following samples do not have taxonomy info:'
            print(msg)
            for i in no_taxo_code:
                print(i)
        return taxo_code_counter

    def count_samples_per_location_by_taxo_code(self,
                                                sample_length_limit=None,
                                                version='V2'):
        """Go through rows of the excell and count category population

            returns:
                taxo_code_counter: {'0.0.1':{'location_id_1':12,'location_id_1':45 ...} ...}
        """
        no_taxo_code = []
        taxo2loc_dict = {}

        if sample_length_limit is None:
            sample_length_limit = self.min_allowed_sample_length

        for audio_ins in self.store.values():
            if audio_ins.taxo_codes is None:
                no_taxo_code.append(audio_ins)
                continue

            sample_count = audio_ins.sample_count(
                excerpt_length=self.excerpt_length,
                sample_length_limit=sample_length_limit)

            if version == 'V1':
                taxo2loc_dict.setdefault(audio_ins.taxo_codes, Counter({}))

                taxo2loc_dict[audio_ins.taxo_codes] = taxo2loc_dict[
                    audio_ins.taxo_codes] + Counter(
                        {audio_ins.location_id: sample_count})

            elif version == 'V2':
                for taxonomy_code in audio_ins.taxo_codes:
                    taxo2loc_dict.setdefault(taxonomy_code, Counter({}))

                    taxo2loc_dict[taxonomy_code] = taxo2loc_dict[
                        taxonomy_code] + Counter({
                            '_'.join([audio_ins.region, audio_ins.location_id]):
                                sample_count
                        })
        if no_taxo_code:
            msg = 'following samples do not have taxonomy info:'
            print(msg)
            for i in no_taxo_code:
                print(i)
        return taxo2loc_dict

    def dataset_generate_samples(self, excerpt_len):
        '''divida into chunks by expected_len seconds.
            Repeats data if smaller than expected_len.

        '''
        for sound_ins in self.values():
            if len(sound_ins.data.shape) == 2:
                # data_to_samples(sound_ins, excerpt_len)
                sound_ins.data_to_samples(excerpt_len=excerpt_len)
            elif len(sound_ins.data.shape) == 1:
                sound_ins.data = sound_ins.data.reshape(1, -1)
                sound_ins.data_to_samples(excerpt_len=excerpt_len)
                sound_ins.data = sound_ins.data.reshape(-1)
            else:
                raise Exception('data shape not supported')
        return self
