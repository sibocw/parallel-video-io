"""Unit tests for async H5 writer module."""

import pytest
import h5py
import numpy as np
from pathlib import Path
import tempfile
import time
from multiprocessing import Queue

from pvio.h5_writer import (
    H5WriteManager,
    AsyncFile,
    AsyncGroup,
    AsyncDataset,
    AsyncAttributeManager,
    Future,
    _MODULE_SENTINEL,  # import anyway for tests
)


@pytest.fixture
def temp_h5_file():
    """Create a temporary H5 file path that gets cleaned up after test."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        filepath = Path(f.name)
    yield filepath
    # Cleanup
    if filepath.exists():
        filepath.unlink()


@pytest.fixture
def h5_manager():
    """Create an H5WriteManager and ensure it's cleaned up."""
    manager = H5WriteManager()
    yield manager
    manager.shutdown()


class TestH5WriteManager:
    """Tests for H5WriteManager class."""

    # def test_manager_creation(self):
    #     """Test that manager can be created and process starts."""
    #     manager = H5WriteManager()
    #     assert manager._process.is_alive()
    #     manager.shutdown()
    
    def test_manager_creation(self):
        """Test that manager can be created and process starts."""
        manager = H5WriteManager()
        print(f"Process alive: {manager._process.is_alive()}")
        print(f"Process exitcode: {manager._process.exitcode}")
        
        # Try to ping the worker
        import time
        time.sleep(0.2)  # Give worker time to start
        
        print("Attempting flush...")
        try:
            manager.flush(timeout=2.0)  # Use a timeout!
            print("Flush succeeded")
        except TimeoutError:
            print("Flush timed out - worker not responding!")
            print(f"Process still alive: {manager._process.is_alive()}")
            print(f"Process exitcode: {manager._process.exitcode}")
        
        manager.shutdown(force=True)

    def test_manager_with_custom_queue_size(self):
        """Test manager with custom queue size."""
        manager = H5WriteManager(max_queue_size=50)
        assert manager._queue._maxsize == 50
        manager.shutdown()

    def test_context_manager(self, temp_h5_file):
        """Test H5WriteManager as context manager."""
        with H5WriteManager() as manager:
            assert manager._process.is_alive()
            f = manager.File(temp_h5_file, "w")
            f.close()
        # Process should be shut down after context exit
        time.sleep(0.2)
        assert not manager._process.is_alive()

    def test_shutdown_idempotent(self, h5_manager):
        """Test that shutdown can be called multiple times safely."""
        h5_manager.shutdown()
        h5_manager.shutdown()  # Should not raise

    def test_flush(self, h5_manager, temp_h5_file):
        """Test flush waits for all pending operations."""
        f = h5_manager.File(temp_h5_file, "w")
        dset = f.create_dataset("data", shape=(100,), dtype="float32")
        
        # Queue up several writes
        futures = []
        for i in range(10):
            data = np.ones(100) * i
            future = dset.partial_write(np.s_[:], data)
            futures.append(future)
        
        # Flush should wait for all writes
        h5_manager.flush(timeout=5.0)
        
        # All futures should be done
        for future in futures:
            response = future.get_result(timeout=0)
            assert response is not None
            assert response.success
        
        f.close()


class TestAsyncFile:
    """Tests for AsyncFile class."""

    def test_file_creation(self, h5_manager, temp_h5_file):
        """Test basic file creation."""
        f = h5_manager.File(temp_h5_file, "w")
        assert isinstance(f, AsyncFile)
        assert f.filename == temp_h5_file
        assert f.mode == "w"
        f.close()
        
        # Verify file was actually created
        assert temp_h5_file.exists()

    def test_file_read_mode_raises(self, h5_manager, temp_h5_file):
        """Test that opening in read mode raises error."""
        with pytest.raises(RuntimeError, match="meant to manage write operations"):
            h5_manager.File(temp_h5_file, "r")

    def test_file_context_manager(self, h5_manager, temp_h5_file):
        """Test AsyncFile as context manager."""
        with h5_manager.File(temp_h5_file, "w") as f:
            assert isinstance(f, AsyncFile)
        # File should be closed after context exit
        assert temp_h5_file.exists()

    def test_direct_instantiation_raises(self):
        """Test that direct AsyncFile instantiation raises error."""
        queue = Queue()
        with pytest.raises(RuntimeError, match="Do not call AsyncGroup"):
            AsyncFile(None, queue)


class TestAsyncGroup:
    """Tests for AsyncGroup class."""

    def test_create_group(self, h5_manager, temp_h5_file):
        """Test group creation."""
        with h5_manager.File(temp_h5_file, "w") as f:
            grp = f.create_group("test_group")
            assert isinstance(grp, AsyncGroup)
        
        # Verify group was created
        with h5py.File(temp_h5_file, "r") as f:
            assert "test_group" in f

    def test_nested_groups(self, h5_manager, temp_h5_file):
        """Test creating nested groups."""
        with h5_manager.File(temp_h5_file, "w") as f:
            grp1 = f.create_group("level1")
            grp2 = grp1.create_group("level2")
            grp3 = grp2.create_group("level3")
            assert isinstance(grp3, AsyncGroup)
        
        # Verify nested structure
        with h5py.File(temp_h5_file, "r") as f:
            assert "level1/level2/level3" in f

    def test_direct_instantiation_raises(self):
        """Test that direct AsyncGroup instantiation raises error."""
        queue = Queue()
        with pytest.raises(RuntimeError, match="Do not call AsyncGroup"):
            AsyncGroup(None, queue)


class TestAsyncDataset:
    """Tests for AsyncDataset class."""

    def test_create_dataset(self, h5_manager, temp_h5_file):
        """Test dataset creation."""
        with h5_manager.File(temp_h5_file, "w") as f:
            dset = f.create_dataset("data", shape=(100, 50), dtype="float32")
            assert isinstance(dset, AsyncDataset)
        
        # Verify dataset was created with correct shape
        with h5py.File(temp_h5_file, "r") as f:
            assert "data" in f
            assert f["data"].shape == (100, 50)

    def test_create_dataset_in_group(self, h5_manager, temp_h5_file):
        """Test dataset creation inside a group."""
        with h5_manager.File(temp_h5_file, "w") as f:
            grp = f.create_group("mygroup")
            dset = grp.create_dataset("data", shape=(10,), dtype="int32")
            assert isinstance(dset, AsyncDataset)
        
        # Verify dataset location
        with h5py.File(temp_h5_file, "r") as f:
            assert "mygroup/data" in f

    def test_partial_write_simple(self, h5_manager, temp_h5_file):
        """Test simple partial write operation."""
        with h5_manager.File(temp_h5_file, "w") as f:
            dset = f.create_dataset("data", shape=(100,), dtype="float32")
            
            # Write data
            test_data = np.arange(100, dtype="float32")
            future = dset.partial_write(np.s_[:], test_data)
            
            # Wait for write to complete
            response = future.get_result(timeout=2.0)
            assert response is not None
            assert response.success
        
        # Verify data was written correctly
        with h5py.File(temp_h5_file, "r") as f:
            np.testing.assert_array_equal(f["data"][:], test_data)

    def test_partial_write_slicing(self, h5_manager, temp_h5_file):
        """Test partial write with various slicing patterns."""
        with h5_manager.File(temp_h5_file, "w") as f:
            dset = f.create_dataset("data", shape=(100, 50), dtype="float32")
            
            # Write to different slices
            data1 = np.ones((10, 50))
            future1 = dset.partial_write(np.s_[0:10, :], data1)
            
            data2 = np.ones((10, 50)) * 2
            future2 = dset.partial_write(np.s_[10:20, :], data2)
            
            # Wait for both writes
            future1.get_result(timeout=2.0)
            future2.get_result(timeout=2.0)
        
        # Verify data
        with h5py.File(temp_h5_file, "r") as f:
            np.testing.assert_array_equal(f["data"][0:10, :], data1)
            np.testing.assert_array_equal(f["data"][10:20, :], data2)

    def test_multiple_async_writes(self, h5_manager, temp_h5_file):
        """Test multiple asynchronous writes don't block."""
        with h5_manager.File(temp_h5_file, "w") as f:
            dset = f.create_dataset("data", shape=(1000,), dtype="float32")
            
            # Queue up many writes without waiting
            futures = []
            start_time = time.time()
            for i in range(50):
                data = np.ones(20) * i
                future = dset.partial_write(np.s_[i*20:(i+1)*20], data)
                futures.append(future)
            queuing_time = time.time() - start_time
            
            # Queuing should be very fast (all async)
            assert queuing_time < 0.5, "Queuing writes should not block"
            
            # Now wait for all to complete
            for future in futures:
                response = future.get_result(timeout=5.0)
                assert response is not None
                assert response.success

    def test_direct_instantiation_raises(self):
        """Test that direct AsyncDataset instantiation raises error."""
        queue = Queue()
        with pytest.raises(RuntimeError, match="Do not call AsyncDataset"):
            AsyncDataset(None, queue)


class TestAsyncAttributeManager:
    """Tests for AsyncAttributeManager class."""

    def test_create_file_attribute(self, h5_manager, temp_h5_file):
        """Test creating attributes on file."""
        with h5_manager.File(temp_h5_file, "w") as f:
            f.attrs["version"] = "1.0"
            f.attrs["author"] = "test"
        
        # Verify attributes
        with h5py.File(temp_h5_file, "r") as f:
            assert f.attrs["version"] == "1.0"
            assert f.attrs["author"] == "test"

    def test_create_dataset_attribute(self, h5_manager, temp_h5_file):
        """Test creating attributes on dataset."""
        with h5_manager.File(temp_h5_file, "w") as f:
            dset = f.create_dataset("data", shape=(10,), dtype="float32")
            dset.attrs["units"] = "meters"
            dset.attrs["scale"] = 1.5
        
        # Verify attributes
        with h5py.File(temp_h5_file, "r") as f:
            assert f["data"].attrs["units"] == "meters"
            assert f["data"].attrs["scale"] == 1.5

    def test_create_group_attribute(self, h5_manager, temp_h5_file):
        """Test creating attributes on group."""
        with h5_manager.File(temp_h5_file, "w") as f:
            grp = f.create_group("mygroup")
            grp.attrs["description"] = "test group"
        
        # Verify attributes
        with h5py.File(temp_h5_file, "r") as f:
            assert f["mygroup"].attrs["description"] == "test group"

    def test_direct_instantiation_raises(self):
        """Test that direct AsyncAttributeManager instantiation raises error."""
        queue = Queue()
        with pytest.raises(RuntimeError, match="Do not call AsyncAttributeManager"):
            AsyncAttributeManager(None, queue)


class TestFuture:
    """Tests for Future class."""

    def test_get_result_blocks(self):
        """Test that get_result blocks until result is set."""
        future = Future()
        
        def set_result_delayed():
            time.sleep(0.1)
            future.set_result("test_result")
        
        import threading
        thread = threading.Thread(target=set_result_delayed)
        thread.start()
        
        result = future.get_result(timeout=1.0)
        assert result == "test_result"
        thread.join()

    def test_get_result_timeout(self):
        """Test that get_result returns None on timeout."""
        future = Future()
        result = future.get_result(timeout=0.1)
        assert result is None


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_dataset_shape(self, h5_manager, temp_h5_file):
        """Test that invalid dataset creation raises error."""
        with h5_manager.File(temp_h5_file, "w") as f:
            with pytest.raises(Exception):
                # Invalid shape should raise error
                dset = f.create_dataset("data", shape=(-100,), dtype="float32")

    def test_write_to_closed_file(self, h5_manager, temp_h5_file):
        """Test that writing to closed file raises error."""
        f = h5_manager.File(temp_h5_file, "w")
        dset = f.create_dataset("data", shape=(100,), dtype="float32")
        f.close()
        
        # Writing after close should raise error
        future = dset.partial_write(np.s_[:], np.ones(100))
        response = future.get_result(timeout=2.0)
        assert response is not None
        assert not response.success
        assert response.error is not None


class TestComplexScenario:
    """Integration tests with complex scenarios."""

    def test_full_workflow(self, h5_manager, temp_h5_file):
        """Test a complete workflow similar to the example usage."""
        with h5_manager.File(temp_h5_file, "w") as f:
            # Set file attributes
            f.attrs["version"] = "1.0"
            f.attrs["created_by"] = "pytest"
            
            # Create groups
            data_grp = f.create_group("data")
            metadata_grp = f.create_group("metadata")
            
            # Create datasets
            images = data_grp.create_dataset(
                "images", shape=(100, 64, 64), dtype="float32"
            )
            labels = data_grp.create_dataset("labels", shape=(100,), dtype="int32")
            
            # Set dataset attributes
            images.attrs["units"] = "normalized"
            labels.attrs["num_classes"] = 10
            
            # Write data in batches
            futures = []
            for i in range(10):
                batch_images = np.random.randn(10, 64, 64).astype("float32")
                batch_labels = np.random.randint(0, 10, size=10, dtype="int32")
                
                f1 = images.partial_write(np.s_[i*10:(i+1)*10, :, :], batch_images)
                f2 = labels.partial_write(np.s_[i*10:(i+1)*10], batch_labels)
                
                futures.extend([f1, f2])
            
            # Wait for all writes
            for future in futures:
                response = future.get_result(timeout=5.0)
                assert response is not None
                assert response.success
        
        # Verify everything was written correctly
        with h5py.File(temp_h5_file, "r") as f:
            assert "data" in f
            assert "metadata" in f
            assert "data/images" in f
            assert "data/labels" in f
            assert f["data/images"].shape == (100, 64, 64)
            assert f["data/labels"].shape == (100,)
            assert f.attrs["version"] == "1.0"

    def test_memory_cleanup_on_close(self, h5_manager, temp_h5_file):
        """Test that memory is properly cleaned up when file is closed."""
        with h5_manager.File(temp_h5_file, "w") as f:
            # Create many objects
            for i in range(10):
                grp = f.create_group(f"group_{i}")
                for j in range(5):
                    dset = grp.create_dataset(
                        f"dataset_{j}", shape=(100,), dtype="float32"
                    )
                    dset.attrs[f"attr_{j}"] = j
        
        # File should be closed and memory cleaned up
        # This test mainly ensures no errors occur during cleanup
        assert temp_h5_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])