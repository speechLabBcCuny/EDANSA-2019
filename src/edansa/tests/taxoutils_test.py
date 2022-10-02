import pytest
from edansa import taxoutils

test_data_megan_excell_row2yaml_code = [
    (
        {
            'Anthro/Bio': '',
            'Category': 'Bug',
            'Specific Category': '',
        },
        'V1',
        '1.3.0',
    ),
    (
        {
            'Anthro/Bio': 'TEST',
            'Category': 'BUg',
            'Specific Category': '',
        },
        'V1',
        '1.3.0',
    ),
    (
        {
            'Anthro/Bio': 'antH',
            'Category': '',
            'Specific Category': '',
        },
        'V1',
        '0.0.0',
    ),
]


@pytest.mark.parametrize('row,version, expected_output',
                         test_data_megan_excell_row2yaml_code)
def test_megan_excell_row2yaml_code(row, expected_output, version):
    output = taxoutils.megan_excell_row2yaml_code(row, version=version)

    assert expected_output == output


testv2_data_megan_excell_row2yaml_code = [
    (
        {
            'Anthro/Bio': '',
            'Category': 'Bug',
            'Specific Category': '',
        },
        'V2',
        [],
    ),
    (
        {
            'bird': '1',
        },
        'V2',
        ['1.1.0'],
    ),
    (
        {
            'Bio': 1,
            'bird': '1',
            'flare': '0',
        },
        'V2',
        [
            '1.0.0',
            '1.1.0',
        ],
    ),
]


# testing version 2 of the same function
@pytest.mark.parametrize('row,version, expected_output',
                         testv2_data_megan_excell_row2yaml_code)
def testv2_megan_excell_row2yaml_code(row, expected_output, version):
    output = taxoutils.megan_excell_row2yaml_code(row, version=version)

    assert expected_output == output
