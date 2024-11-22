from dask.distributed import Client, print as dprint
import dask


def get_dask_client() -> dask.distributed.Client:
    return Client.current()


async def compute(client: Client, tasks):
    """Calls client.compute"""
    if client.asynchronous:
        if isinstance(tasks, (list, set, tuple)):
            futures = client.compute(tasks)
            result = await client.gather(futures)
        else:  # somehow, this is a special case that needs to be handled separately
            result = await client.compute(tasks).result()
    else:
        result = client.compute(tasks, sync=True)
    return result


async def dask_url_io_test(client: Client, url: str):
    """Test the functionality of the client in read and write from url.

    URL provided must be writable and allow testing purpose;
    If using a synchronous client, then call this with client.sync(client_url_test)

    Args:
        client: The client to test on
        url: The url to test on
    """
    import numpy as np
    import dask.array as da
    from cvpl_tools.fsspec import RDirFileSystem
    import cvpl_tools.ome_zarr.io as ome_io
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    def log(client: dask.distributed.Client, msg, end='\n'):
        dprint(msg, end=end)
        client.log_event(topic='dask_url_io_test', msg=msg)
        logger.error(msg)

    dprint(f'-----------------------Testing url {url}-----------------------')

    dask_arr = da.from_array(np.arange(16, dtype=np.float32).reshape((4, 4)))

    log(client, f'Testing await client.submit(da.Array.compute, dask_arr):', end='')
    result = await client.submit(da.Array.compute, dask_arr)
    np.testing.assert_almost_equal(result.sum(), 120.)
    log(client, 'ok.')

    log(client, f'Testing await client.compute(dask_arr).result():', end='')
    result = await client.compute(dask_arr).result()
    np.testing.assert_almost_equal(result.sum(), 120.)
    log(client, 'ok.')

    i = 0
    while True:
        TMP_DIR = f'{url}/tmp{i}'
        log(client, f'Testing if path {TMP_DIR} exists:', end='')
        tmpfs = RDirFileSystem(TMP_DIR)
        if tmpfs.exists(path=''):
            log(client, 'Exists.')
            i += 1
        else:
            log(client, 'Does not exists, new files will be created under this path. ', end='')
            break

    log(client, f'Creating new folder at path {TMP_DIR}:', end='')
    tmpfs.ensure_dir_exists(remove_if_already_exists=False)
    log(client, f'ok.')
    log(client, f'Verifying the newly created folder exists:', end='')
    assert tmpfs.exists(path=''), f'Folder does not exist!'
    log(client, f'ok.')

    txtfs = RDirFileSystem(f'{TMP_DIR}/test.txt')
    log(client, f'Test writing string to path {txtfs.url}:', end='')
    with txtfs.open(path='', mode='w') as outfile:
        outfile.write('This is a test message.')
    log(client, f'ok.')
    log(client, f'Verifying the newly created txt file exists:', end='')
    assert txtfs.exists(path=''), f'File does not exist!'
    log(client, f'ok.')
    log(client, f'Test reading string to path {txtfs.url}:', end='')
    with txtfs.open(path='', mode='r') as infile:
        msg = infile.read()
        assert msg == 'This is a test message.', msg
    log(client, f'ok.')

    omefs = RDirFileSystem(f'{TMP_DIR}/testim.ome.zarr')
    log(client, f'Test writing dask array as ome zarr image at path {omefs.url}:', end='')
    delayed = [da.to_zarr(
        arr=dask_arr,
        url=omefs.url,
        component='0',
        # storage_options=options,
        compute=False,
        # compressor=options.get("compressor", zarr.storage.default_compressor),
        # dimension_separator=group._store._dimension_separator,
    )]
    computed = client.compute(delayed)
    gathered = await client.gather(computed)
    log(client, f'ok.')
    log(client, f'Test reading dask array from ome zarr image just written:', end='')
    read_arr = await ome_io.load_dask_array_from_path(f'{omefs.url}/0', mode='r')
    result = await client.submit(da.Array.compute, read_arr)
    np.testing.assert_almost_equal(result.sum(), 120.)
    log(client, f'ok.')

    omefs = RDirFileSystem(f'{TMP_DIR}/testim2.ome.zarr')
    log(client, f'Test writing dask array as ome zarr image at path {omefs.url}:', end='')
    await ome_io.write_ome_zarr_image(out_ome_zarr_path=omefs.url, da_arr=dask_arr, asynchronous=True)
    log(client, f'ok.')
    log(client, f'Test reading dask array from ome zarr image just written:', end='')
    read_arr = await ome_io.load_dask_array_from_path(f'{omefs.url}/0', mode='r')
    result = await client.submit(da.Array.compute, read_arr)
    np.testing.assert_almost_equal(result.sum(), 120.)
    log(client, f'ok.')

    log(client, f'Test ls the tmp directory:', end='')
    log(client, str(tmpfs.ls(path='')))
    log(client, 'ls completes.')

    log(client, f'Test removing the temporary directory at {TMP_DIR}:', end='')
    tmpfs.rm(path='', recursive=True)
    assert not tmpfs.exists(path=''), 'Folder still exists after attempting to remove!'
    log(client, f'ok.')

    log(client, f'All testing completed for this client-url pair.')


