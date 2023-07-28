from pytorch_lightning.callbacks import Callback



class ClearmlMetricLogger(Callback):
    # TODO handle possible dict loss/metric results

    def on_train_start(self, trainer, pl_module):
        ...
        # TODO Check for `metrics` attribute
        # TODO Check for `loss_fn` attribute
        # TODO Make sure train/val metrics are distinct copies

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        ...
        # NOTE: The value outputs["loss"] here will be the normalized value w.r.t
        # accumulate_grad_batches of the loss returned from training_step.
        # TODO log loss results
        # TODO log metrics update results if compute_on_batch is True

    def on_train_epoch_end(self, trainer, pl_module):
        ...
        # TODO log metrics compute results

    def on_test_start(self, trainer, pl_module):
        ...
        # TODO Check for `metrics` attribute
        # TODO Check for `loss_fn` attribute

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        ...
        # NOTE: The value outputs["loss"] here will be the normalized value w.r.t
        # accumulate_grad_batches of the loss returned from training_step.
        # TODO log loss results
        # TODO log metrics update results if compute_on_batch is True

    def on_test_epoch_end(self, trainer, pl_module):
        ...
        # TODO log metrics compute results
