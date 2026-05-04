import os
import sys
import concurrent.futures
import pytest
import numpy as np
from unittest.mock import MagicMock

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, BACKEND_DIR)

from services.data_loader import load_step_data  # noqa: E402
from services.storage_io import write_zarr_group, zarr_available  # noqa: E402

@pytest.fixture
def mock_cache():
    from collections import OrderedDict
    return OrderedDict()

@pytest.fixture
def mock_data_dir(tmp_path):
    return str(tmp_path)

def test_singleflight_concurrent(mock_cache, mock_data_dir):
    if not zarr_available():
        pytest.skip("zarr is not installed")

    run_dir = os.path.join(mock_data_dir, 'icon-d2', '2024022000')
    os.makedirs(run_dir, exist_ok=True)
    assert write_zarr_group(
        os.path.join(run_dir, '001.zarr'),
        {
            'lat': np.arange(10, dtype=np.float32),
            'lon': np.arange(10, dtype=np.float32),
            'ww': np.zeros((10, 10), dtype=np.float32),
        },
    )
    
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
    
    # All concurrent loads should share one cache entry (singleflight)
    # Verify cache filled
    assert len(mock_cache) == 1
    cached = list(mock_cache.values())[0]
    assert 'ww' in cached
    assert 'lat' in cached
    assert 'lon' in cached
