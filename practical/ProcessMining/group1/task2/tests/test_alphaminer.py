import pytest
import pm4py
import pandas as pd
from practical.ProcessMining.group1.task2.alphaminer import AlphaMiner
from typing import Dict, Any

FILE_PATH = '../example_files/common-example.csv'  # event log from the paper


@pytest.fixture
def alpha_miner() -> AlphaMiner:
    return AlphaMiner(FILE_PATH)


@pytest.fixture
def event_log() -> pd.DataFrame:
    event_log = pd.read_csv(FILE_PATH, sep=';')
    return pm4py.format_dataframe(event_log, case_id='case_id', activity_key='activity', timestamp_key='timestamp')


@pytest.fixture
def footprints(event_log: pd.DataFrame) -> Dict[str, Any]:
    footprints = pm4py.discovery.discover_footprints(event_log)
    # convert to set (discard frequency information of dictionary as our implementation does not provide that)
    footprints['dfg'] = set(footprints['dfg'].keys())
    return footprints


def test_footprints_discovery(alpha_miner: AlphaMiner, footprints: Dict[str, Any]) -> None:
    # test if discovered footprints match PM4PY (ignoring frequency information)
    assert alpha_miner.discover_footprints() == footprints
