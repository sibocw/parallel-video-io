# Sequential GPU decode with DALI.
from nvidia.dali import fn, pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


def make_loader(path, batch_size=16):
    @pipeline_def(batch_size=batch_size, num_threads=2, device_id=0)
    def pipe():
        return fn.readers.video(
            device="gpu", filenames=[path], sequence_length=1, name="reader"
        )

    p = pipe()
    p.build()
    return DALIGenericIterator(
        [p], ["frames"], reader_name="reader", last_batch_policy=LastBatchPolicy.PARTIAL
    )
