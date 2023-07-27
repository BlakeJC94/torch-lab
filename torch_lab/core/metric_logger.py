from pytorch_lightning.callbacks import Callback

# On init, check attribtes and suplicate metics for stages
# On train batch/epoch ends, log metrics located my module attributes if they're there
class ClearmlMetricLogger(Callback):
    ...
