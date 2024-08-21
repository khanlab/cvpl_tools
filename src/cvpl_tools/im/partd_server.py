import zmq
import logging
import socket
from contextlib import contextmanager, suppress
import os
from threading import Thread, Lock
import shutil

from partd.dict import Dict
from partd.buffer import Buffer
import sqlite3
from partd.core import Interface
import locket


class SQLiteKVStore:
    def __init__(self, db_path: str):
        self.is_exists = os.path.exists(db_path)  # may not be accurate
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        if not self.is_exists:
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS kv_store (
                id TEXT PRIMARY KEY,
                value TEXT
            )
            ''')

    def write_rows(self, tups):
        self.cursor.executemany('''
        INSERT INTO kv_store (id, value) VALUES (?, ?)
        ON CONFLICT(id) DO UPDATE SET value=excluded.value;
        ''', tups)

    def write_row(self, tup):
        self.cursor.execute('''
        INSERT INTO kv_store (id, value) VALUES (?, ?)
        ON CONFLICT(id) DO UPDATE SET value=excluded.value;
        ''', tup)

    def read_rows(self, ids):
        return [self.read_row(id) for id in ids]

    def read_row(self, id):
        self.cursor.execute('''
        SELECT value FROM kv_store WHERE id = ?
        ''', (id,))
        return self.cursor.fetchone()[0]

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()


class SQLitePartd(Interface):
    """reference: https://github.com/dask/partd/blob/main/partd/file.py

    commit hash: efa78b4
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.kv_store = SQLiteKVStore(f'{path}/sqlite.db')
        self.lock = locket.lock_file(f'{path}/.lock')
        Interface.__init__(self)

    def __getstate__(self):
        return {'path': self.path}

    def __setstate__(self, state):
        Interface.__setstate__(self, state)
        self.__class__.__init__(self, state['path'])

    def append(self, data, lock=True, fsync=False, **kwargs):
        if lock:
            self.lock.acquire()
        try:
            self.kv_store.write_rows(data.items())
            self.kv_store.commit()
        finally:
            if lock:
                self.lock.release()

    def _get(self, keys, lock=True, **kwargs):
        assert isinstance(keys, (list, tuple, set))
        if lock:
            self.lock.acquire()
        try:
            if len(keys) == 1:
                result = [self.kv_store.read_row(keys[0])]
            else:
                result = self.kv_store.read_rows(keys)
        finally:
            if lock:
                self.lock.release()
        return result

    def _iset(self, key, value, lock=True):
        """ Idempotent set """
        if lock:
            self.lock.acquire()
        try:
            self.kv_store.write_row((key, value))
            self.kv_store.commit()
        finally:
            if lock:
                self.lock.release()

    def _delete(self, keys, lock=True):
        if lock:
            self.lock.acquire()
        try:
            raise NotImplementedError()
        finally:
            if lock:
                self.lock.release()

    def drop(self):
        self.kv_store.close()
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        self._iset_seen.clear()
        os.mkdir(self.path)
        self.kv_store = SQLiteKVStore(f'{self.path}/sqlite.db')

    def close(self):
        self.kv_store.close()

    def __exit__(self, *args):
        self.drop()
        os.rmdir(self.path)
        self.kv_store.close()


# --------------------------------------------Part 2: Server------------------------------------------------


tuple_sep = b'-|-'


logger = logging.getLogger(__name__)


@contextmanager
def logerrors():
    try:
        yield
    except Exception as e:
        logger.exception(e)
        raise


class Server:
    """Modified from https://github.com/dask/partd/blob/main/partd/zmq.py"""
    def __init__(self, path, bind=None, available_memory=None):
        self.context = zmq.Context()

        self.path = path
        self.available_memory = available_memory

        self.socket = self.context.socket(zmq.ROUTER)

        hostname = socket.gethostname()
        if isinstance(bind, str):
            bind = bind.encode()
        if bind is None:
            port = self.socket.bind_to_random_port('tcp://*')
        else:
            self.socket.bind(bind)
            port = int(bind.split(':')[-1].rstrip('/'))
        self.address = ('tcp://%s:%d' % (hostname, port)).encode()

        self.status = 'created'

        self._lock = Lock()
        self._socket_lock = Lock()

        self.start()

    def get_partd(self):
        partd = SQLitePartd(self.path)
        if self.available_memory is not None:
            buffer_partd = Buffer(Dict(), partd, available_memory=self.available_memory)
        else:
            buffer_partd = partd
        return partd, buffer_partd

    def start(self):
        if self.status != 'run':
            self.status = 'run'
            self._listen_thread = Thread(target=self.listen)
            self._listen_thread.start()
            logger.debug('Start server at %s', self.address)

    def block(self):
        """ Block until all threads close """
        try:
            self._listen_thread.join()
        except AttributeError:
            pass

    def listen(self):
        with logerrors():
            logger.debug('Start listening %s', self.address)
            partd, buffer_partd = self.get_partd()
            while self.status != 'closed':
                if not self.socket.poll(100):
                    continue

                with self._socket_lock:
                    payload = self.socket.recv_multipart()

                address, command, payload = payload[0], payload[1], payload[2:]
                logger.debug('Server receives %s %s', address, command)
                if command == b'close':
                    logger.debug('Server closes')
                    self.ack(address)
                    self.status = 'closed'
                    break
                    # self.close()

                elif command == b'append':
                    keys, values = payload[::2], payload[1::2]
                    keys = list(map(deserialize_key, keys))
                    data = dict(zip(keys, values))
                    buffer_partd.append(data, lock=False)
                    logger.debug('Server appends %d keys', len(data))
                    self.ack(address)

                elif command == b'iset':
                    key, value = payload
                    key = deserialize_key(key)
                    buffer_partd.iset(key, value, lock=False)
                    self.ack(address)

                elif command == b'get':
                    keys = list(map(deserialize_key, payload))
                    logger.debug('get %s', keys)
                    result = buffer_partd.get(keys)
                    self.send_to_client(address, result)
                    self.ack(address, flow_control=False)

                elif command == b'syn':
                    self.ack(address)

                # drop and delete commands are not used and not supported

                else:
                    logger.debug("Unknown command: %s", command)
                    raise ValueError("Unknown command: " + command)

            if buffer_partd != partd:
                buffer_partd.flush()
            partd.close()

    def send_to_client(self, address, result):
        with logerrors():
            if not isinstance(result, list):
                result = [result]
            with self._socket_lock:
                self.socket.send_multipart([address] + result)

    def ack(self, address, flow_control=True):
        with logerrors():
            logger.debug('Server sends ack')
            self.send_to_client(address, b'ack')

    def close(self):
        logger.debug('Server closes')
        self.status = 'closed'
        self.block()
        with suppress(zmq.error.ZMQError):
            self.socket.close(1)
        with suppress(zmq.error.ZMQError):
            self.context.destroy(3)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()


def deserialize_key(text):
    """
    >>> deserialize_key(b'x')
    b'x'
    >>> deserialize_key(b'a-|-b-|-1')
    (b'a', b'b', b'1')
    """
    if tuple_sep in text:
        return tuple(text.split(tuple_sep))
    else:
        return text
