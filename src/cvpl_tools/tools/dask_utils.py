from dask.distributed import Client
import dask


def get_dask_client() -> dask.distributed.Client:
    return Client.current()


async def compute(client: Client, tasks):
    """Calls client.compute"""
    if client.asynchronous:
        if isinstance(tasks, (list, set, tuple)):
            result = await client.compute(tasks, sync=True)
        else:  # somehow, this is a special case that needs to be handled separately
            result = await client.compute(tasks).result()
    else:
        result = client.compute(tasks, sync=True)
    return result

