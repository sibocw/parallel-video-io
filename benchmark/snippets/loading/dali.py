# Build a parallel multi-video frame loader on the GPU with DALI.
from nvidia.dali import fn, pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


def make_loader(paths, batch_size, num_workers):
    @pipeline_def(batch_size=batch_size, num_threads=num_workers, device_id=0)
    def pipe():
        return fn.readers.video(
            device="gpu", filenames=paths, sequence_length=1, name="reader"
        )

    p = pipe()
    p.build()
    return DALIGenericIterator(
        [p], ["frames"], reader_name="reader", last_batch_policy=LastBatchPolicy.PARTIAL
    )
