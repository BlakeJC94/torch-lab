from torch_lab.modules import LabModule


class TestTrainModuleCheckpoint:
    """Test that TrainModule saves a checkpoint that can be loaded."""

    def test_load_checkpoint_into_raw_model(self, mock_model, train_module_checkpoint):
        """Test that TrainModule checkpoint can be loaded directly into model instance."""
        mock_model.load_state_dict(train_module_checkpoint["state_dict"])

    def test_load_checkpoint_into_lab_module(self, mock_model, train_module_checkpoint):
        """Test that state_dict can be loaded into a LabModule"""
        LabModule(mock_model).model.load_state_dict(
            train_module_checkpoint["state_dict"]
        )

    def test_load_checkpoint_into_train_lab_module(
        self, train_module, train_module_checkpoint
    ):
        """Test that state_dict can be loaded into a TrainLabModule (i.e. load checkpoint weights
        only).
        """
        train_module.model.load_state_dict(train_module_checkpoint["state_dict"])

    def test_load_checkpoint(self, train_module, mock_dataloader, trainer, ckpt_path):
        trainer.validate(
            train_module,
            mock_dataloader,
            ckpt_path=ckpt_path,
        )
