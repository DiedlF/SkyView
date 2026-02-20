import os
import tempfile
import threading
import time
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from data_loader import load_step_data  # adjust import if needed

@pytest.fixture
def mock_cache():
    from collections import OrderedDict
    return OrderedDict()

@pytest.fixture
def mock_data_dir(tmp_path):
    return str(tmp_path)

def test_singleflight_concurrent(mock_cache, mock_data_dir, monkeypatch):
    # Mock npz load
    def mock_np_load(path):
        arr = np.load(path)
        return MagicMock(**{k: np.zeros((10,10)) for k in ['lat', 'lon', 'ww']})
    
    monkeypatch.setattr(np, 'load', mock_np_load)
    
    # Create fake npz path
    npz_path = os.path.join(mock_data_dir, 'icon-d2', '2024022000', '001.npz')
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for i in range(5):
            keys = ['ww'] if i % 2 else ['lat', 'lon']
            future = executor.submit(
                load_step_data,
                data_dir=mock_data_dir,
                model='icon-d2',
                run='2024022000',
                step=1,
                cache=mock_cache,
                cache_max_items=10,
                keys=keys,
                logger=MagicMock(),
            )
            futures.append(future)
    
    results = [f.result() for f in futures]
    
    # All concurrent loads should share 1 NPZ load (singleflight)
    # Verify cache filled
    assert len(mock_cache) == 1
    cached = list(mock_cache.values())[0]
    assert 'ww' in cached
    assert 'lat' in cached
    assert 'lon' in cached