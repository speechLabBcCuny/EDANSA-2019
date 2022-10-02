'''Tests for dataimport module.

'''

from edansa import dataimport
import pytest
import numpy as np
import torch

test_data_data_to_samples = [
    (10, 54, 1, 48000.0, 'random/path/to/audio/file'),
    (10, 5.1, 1, 48000.0, 'random/path/to/audio/file'),
    (10.0, 5.1, 1, 48000.0, 'random/path/to/audio/file'),
    (10.0, 58.1, 1, 48000, 'random/path/to/audio/file'),
    (10.0, 54.1, 1, 48000, 'random/path/to/audio/file'),
    (20.0, 54.1, 1, 48000, 'random/path/to/audio/file'),
    (10.0, 54.1, 2, 48000, 'random/path/to/audio/file'),
    (10.0, 56.1, 2, 48000, 'random/path/to/audio/file'),
]


@pytest.mark.parametrize(
    'excerpt_len,audio_file_len,channel_count,audio_sr,audio_file_path',
    test_data_data_to_samples,
    # indirect=True,
)
def test_data_to_samples(excerpt_len, audio_file_len, channel_count, audio_sr,
                         audio_file_path):
    print(excerpt_len, audio_file_len, channel_count, audio_sr, audio_file_path)
    # excerpt_len,audio_file_len,audio_sr,audio_file_path  = inputs

    sound_ins = dataimport.Audio(audio_file_path, audio_file_len)
    sound_ins.sr = audio_sr
    if channel_count == 1:
        sound_ins.data = np.ones((1, int(audio_file_len * audio_sr)))
    else:
        sound_ins.data = np.ones(
            (channel_count, int(audio_file_len * audio_sr)))

    sound_ins.data_to_samples(excerpt_len=excerpt_len)

    assert isinstance(sound_ins.samples, list)
    if audio_file_len > 10:
        sample_count = audio_file_len // excerpt_len
        trim_point = int(sample_count * excerpt_len * sound_ins.sr)
        if audio_file_len % excerpt_len >= 5:
            # test if after trim_point is all zeros
            assert np.sum(sound_ins.samples[-1][trim_point:]) == 0
            sample_count += 1
        # test that the number of samples is correct
        assert len(sound_ins.samples) == sample_count
    else:
        assert len(sound_ins.samples) == 1
    if channel_count == 1:
        assert np.sum(sound_ins.samples[0][int(audio_file_len *
                                               audio_sr):]) == 0
    else:
        assert np.sum(sound_ins.samples[0][:,
                                           int(audio_file_len *
                                               audio_sr):]) == 0
    assert sound_ins.samples[0].size == audio_sr * excerpt_len * channel_count

    bb = np.array(sound_ins.samples)
    _ = torch.from_numpy(bb)
