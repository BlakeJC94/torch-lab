import torch


class TestBaseDataset:
    def test_getitem(self, mock_dataset, n_samples):
        """Test that transforms are applied and all metadata is returned."""
        for i, (x, md) in enumerate(mock_dataset):
            assert (x == 2 * i * torch.ones_like(x)).all()
            assert md["i"] == i
            assert md["y"] == i % 2
            assert md["foo"] == "bar"

    def test_getitem_predict(self, mock_predict_dataset, n_samples):
        """Test that metadata excludes key "y" from metadata if get_raw_label returns None."""
        for i, (x, md) in enumerate(mock_predict_dataset):
            assert (x == 2 * i * torch.ones_like(x)).all()
            assert md["i"] == i
            assert md["foo"] == "bar"
            assert "y" not in md
