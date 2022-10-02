"""Utilities for clipping detection in audio.

'Clipping is a form of waveform distortion that occurs when an amplifier is
overdriven and attempts to deliver an output voltage or current beyond its
maximum capability.'(wikipedia)
Clipped samples of the audio signal does not carry any information.
We assume clipping happens when sample's value is +1 or -1 (threshold).


    Typical usage    
        Function run_task_save combines load_audio and get_clipping_percent.

    1)example by using list of files :
    ```[python]
    from edansa import clippingutils
    test_area_files = ['./data/sound_examples/10minutes.mp3', 
                        './data/sound_examples/10seconds.mp3']

    all_results_dict, files_w_errors = clippingutils.run_task_save(
        test_area_files, "test_area", "./output_folder_path", 1.0)
    ```

    2) example by using a file_properties file :
    ```[python]
    from edansa import clippingutils
    # file info
    import pandas as pd
    clipping_threshold=1.0
    file_properties_df_path = "../nna/data/prudhoeAndAnwr4photoExp_dataV1.pkl"
    file_properties_df = pd.read_pickle(file_properties_df_path)
    # where to save results
    clipping_results_path="./clipping_results/"
    location_ids=['11','12']
    for i,location_id in enumerate(location_ids):
        print(location_id,i)
        location_id_filtered=file_properties_df[file_properties_df.locationId==location_id]
        all_results_dict, files_w_errors = clippingutils.run_task_save(
                                                    location_id_filtered.index,
                                                    location_id,clipping_results_path,
                                                    clipping_threshold)
    ```
"""

from argparse import ArgumentError
import pickle
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np


def load_audio(
    filepath: Union[Path, str],
    dtype: np.dtype = np.int16,
    backend: str = "pydub",
    resample_rate=None,
) -> Tuple[np.ndarray, int]:
    """Load audio file as numpy array using given backend.

    Depending on audio reading library handles data type conversions.

    Args:
        filepath: path to the file
        dtype: Data type to store audio file.
        backend: Which backend to use load the audio.

    Returns:
        A tuple of array storing audio and sampling rate.

    """
    if dtype not in (np.int16, np.float32, np.float64):
        raise TypeError("dtype for loading audio should be one of the: " +
                        "np.int16,np.float32, np.float64")

    filepath = str(filepath)
    if backend == "librosa":
        if resample_rate is not None:
            raise ArgumentError("librosa backend does not support resampling")
        import librosa
        if dtype in (np.float32, np.float64):
            sound_array, sr = librosa.load(filepath,
                                           mono=False,
                                           sr=None,
                                           dtype=dtype)
        # "Librosa does not support integer-valued samples
        #   because many of the downstream analyses (STFT etc) would implicitly
        #   cast to floating point anyway, so we opted to put that requirement
        #   up front in the audio buffer validation check."
        #   src: https://github.com/librosa/librosa/issues/946#issuecomment-520555138

        # convert to int16/PCM-16:
        # 16-bit signed integers, range from −32768 to 32767
        # and librosa returns a np.float32 and normalizes [-1,1]
        #
        # related librosa int16: https://stackoverflow.com/a/53656547/3275163
        if dtype == np.int16:
            sound_array, sr = librosa.load(filepath, mono=False, sr=None)
            sound_array = sound_array.T
            sound_array = sound_array.reshape(-1, 2)
            maxv = np.iinfo(np.int16).max + 1  #(32768)
            sound_array = (sound_array * maxv).astype(np.int16)

            # sound_array, sr = librosa.load(filepath, mono=False, sr=None)
            # sound_array = sound_array * 32768
            # sound_array = sound_array.astype(np.int16)
    elif backend == "pydub":
        from pydub import AudioSegment
        AudioSegment.converter = '/home/enis/sbin/ffmpeg'
        sound_array = AudioSegment.from_file(filepath)
        sr = sound_array.frame_rate
        channels = sound_array.channels
        # print(resample_rate, sr)
        if (resample_rate is not None) and resample_rate != sr:
            sound_array = sound_array.set_frame_rate(resample_rate)
            sr = resample_rate
        sound_array = sound_array.get_array_of_samples()
        sound_array = np.array(sound_array)
        if channels == 2:
            sound_array = sound_array.reshape((-1, 2)).T
        if dtype in (np.float32, np.float64):
            sound_array = sound_array.astype(dtype)
            # this was a BUG, keeping here to understand why I did that
            # sound_array = (sound_array / 32768)
    else:
        raise ValueError(f"no backend called {backend}")
    return sound_array, sr


def get_clipping_percent(sound_array: np.ndarray,
                         threshold: float = 1.0) -> List[np.float64]:
    """Calculate clipping percentage comparing to (>=) threshold.

        Args:
            sound_array: a numpy array with shape of
                        (sample_count) or (2,sample_count)
            threshold: min and max values which samples are assumed to be
                      Clipped  0<=threshold<=1,

    """
    if sound_array.dtype == np.int16:
        minval = int(-32768 * threshold)
        maxval = int(32767 * threshold)
    else:
        minval = -threshold
        #librosa conversion from int to float causes missing precision
        # so we lover it sligtly
        maxval = threshold * 0.9999

    #mono
    if len(sound_array.shape) == 1:
        result = ((
            (np.sum(sound_array <= minval) + np.sum(sound_array >= maxval))) /
                  sound_array.size)
        results = [result]
    elif len(sound_array.shape) == 2:
        results = (np.sum(sound_array <= minval, axis=1) + np.sum(
            sound_array >= maxval, axis=1)) / sound_array.shape[-1]
        results = list(results)
    else:
        raise ValueError("only mono and stereo sound is supported")
    return results


def run_task_save(input_files: List[Union[str, Path]],
                  area_id: str,
                  results_folder: Union[str, Path,],
                  clipping_threshold: float,
                  segment_len: int = 10,
                  audio_load_backend: str = "pydub",
                  save=True) -> Tuple[dict, list]:
    """Save clipping in dict to a file named as f"{area_id}_{threshold}.pkl"

        Computes clipping for only files
         that does not exist in the results pkl file. 

        Args:
            allfiles: List of files to calculate clipping.
            area_id: of the files coming from, will be used in file_name
            results_folder: where to save results if save==True
            clipping_threshold:
            segment_len: length of segments to calculate clipping per.
            audio_load_backend: backend for loading files
            save: Flag for saving results to a file.
        Returns:
            Tuple(all_results_dict ,files_w_errors)
                all_results_dict: Dict{a_file_path:np.array}
                files_w_errors: List[(index, a_file_path, exception),]

    """

    clipping_threshold_str = str(clipping_threshold)
    clipping_threshold_str = clipping_threshold_str.replace(".", ",")
    filename = "{}_{}.pkl".format(area_id, clipping_threshold_str)
    error_filename = "{}_{}_error.pkl".format(area_id, clipping_threshold_str)
    if results_folder:
        results_folder = Path(results_folder)
        results_folder.mkdir(parents=True, exist_ok=True)
        output_file_path = results_folder / filename
        error_file_path = results_folder / error_filename
    else:
        output_file_path = Path('.') / filename
        error_file_path = Path('.') / error_filename

    input_files = [str(i) for i in input_files]

    if results_folder and output_file_path.exists():
        print(f'Clipping file for {filename} exists at {results_folder}.' +
              ' Checking existing results.')
        prev_results_dict = np.load(str(output_file_path), allow_pickle=True)
        prev_results_dict = dict(prev_results_dict[()])
        prev_results_keys = {str(i) for i in prev_results_dict.keys()}
        input_result_keys = set(input_files)
        new_result_keys = input_result_keys.difference(prev_results_keys)
        if len(new_result_keys) > 0:
            print(f'{len(new_result_keys)} number of files missing results' +
                  ', calculating only those.')
        else:
            print('No new file from existing results, will exit.')
        file2process = new_result_keys
        all_results_dict = prev_results_dict
    else:
        file2process = input_files
        all_results_dict = {}

    # return {},[]
    # we need to merge this one as well TODO
    files_w_errors = []

    # CALCULATE RESULTS
    for _, audio_file in enumerate(file2process):
        # try:
        # print(audio_file)
        y, sr = load_audio(audio_file,
                           dtype=np.int16,
                           backend=audio_load_backend)

        assert sr == int(sr)
        sr = int(sr)
        results = []
        # print('yshape', y.shape)
        for clip_i in range(0, int(y.shape[-1] - segment_len),
                            int(segment_len * sr)):
            #mono
            if len(y.shape) == 1:
                res = get_clipping_percent(y[clip_i:(clip_i +
                                                     (segment_len * sr))],
                                           threshold=clipping_threshold)
            elif len(y.shape) == 2:
                res = get_clipping_percent(y[:, clip_i:(clip_i +
                                                        (segment_len * sr))],
                                           threshold=clipping_threshold)
            results.append(res)
        resultsnp = np.array(results)
        all_results_dict[audio_file] = resultsnp[:]
        # except Exception as e:  # pylint: disable=W0703
        # print(i, audio_file)
        # print(e)
        # files_w_errors.append((i, audio_file, e))
    # SAVE RESULTS

    if save:
        with open(output_file_path, "wb") as f:
            np.save(f, all_results_dict)
        if files_w_errors:
            with open(error_file_path, "wb") as f:
                pickle.dump(files_w_errors, f, protocol=pickle.HIGHEST_PROTOCOL)

    return all_results_dict, files_w_errors
