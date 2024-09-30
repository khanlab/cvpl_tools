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
from dask.distributed import print as dprint


class SQLiteKVStore:
    def __init__(self, db_path: str):
        self.is_exists = os.path.exists(db_path)  # may not be accurate
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        assert self.cursor is not None
        self.write_row_stmt = None
        self.read_row_stmt = None
        self.init_db()

    def init_db(self):
        if not self.is_exists:
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS kv_store (
                id TEXT PRIMARY KEY,
                value TEXT
            )
            ''')
        self.write_row_stmt = """
        INSERT INTO kv_store (id, value) VALUES (?, ?)
        ON CONFLICT(id) DO UPDATE SET value=excluded.value;
        """
        self.read_row_stmt = """
        SELECT value FROM kv_store WHERE id = ?
        """

    def write_rows(self, tups):
        self.cursor.executemany(self.write_row_stmt, tups)

    def write_row(self, tup):
        self.cursor.execute(self.write_row_stmt, tup)

    def read_rows(self, ids):
        return [self.read_row(id) for id in ids]

    def read_row(self, id):
        self.cursor.execute(self.read_row_stmt, (id,))
        result = self.cursor.fetchone()

        result = result[0]
        return result

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()


# ------------------------------------------Part 2: Partd Class-----------------------------------------------
"""
Copyright (c) 2015, Continuum Analytics, Inc. and contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

Neither the name of Continuum Analytics nor the names of any contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

SQLitePartd class is modified from the partd class below:
reference: https://github.com/dask/partd/blob/main/partd/file.py commit hash: efa78b4
"""


class SQLitePartd(Interface):
    def __init__(self, path: str, create_kv_store=None):
        """path is an directory under which the lock file will be placed"""
        self.path = path
        os.makedirs(path, exist_ok=True)
        if create_kv_store is None:
            def create_kv_store():
                return SQLiteKVStore(f'{path}/sqlite.db')
        self.kv_store = create_kv_store()
        self.create_kv_store = create_kv_store
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
        self.kv_store = self.create_kv_store()

    def close(self):
        self.kv_store.close()

    def __exit__(self, *args):
        self.drop()
        os.rmdir(self.path)
        self.kv_store.close()


# --------------------------------------------Part 3: Server------------------------------------------------


tuple_sep = b'-|-'


logger = logging.getLogger(__name__)


@contextmanager
def logerrors():
    try:
        yield
    except Exception as e:
        logger.exception(e)
        raise


class SqliteServer:
    """
    SqliteServer class is modified from the Server/Client class file below:
    https://github.com/dask/partd/blob/main/partd/zmq.py
    """
    def __init__(self, path, nappend, available_memory=None, get_sqlite_partd=None):
        """
        Args:
            path:
            available_memory:
            get_sqlite_partd:
            nappend: Number of messages to append, after which point the server closes
        """

        self.path = path
        self.available_memory = available_memory
        if get_sqlite_partd is None:
            def get_sqlite_partd():
                return SQLitePartd(path)
        self._get_sqlite_partd = get_sqlite_partd

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        hostname = socket.gethostname()
        port = self.socket.bind_to_random_port('tcp://*')
        self.address = ('tcp://%s:%d' % (hostname, port)).encode()

        self.status = 'run'
        assert isinstance(nappend, int), f'Expected int, got type(nappend)={type(nappend)}'
        self.nappend = nappend
        self.nappended = 0
        self._socket_lock = Lock()
        self._listen_thread = Thread(target=self.listen)
        self._listen_thread.start()
        logger.debug('Start server at %s', self.address)

    def get_partd(self):
        partd = self._get_sqlite_partd()
        if self.available_memory is not None:
            buffer_partd = Buffer(Dict(), partd, available_memory=self.available_memory)
        else:
            buffer_partd = partd
        return partd, buffer_partd

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
                    self.ack(address)
                    self.close()
                    break

                elif command == b'append':
                    keys, values = payload[::2], payload[1::2]
                    keys = list(map(deserialize_key, keys))
                    data = dict(zip(keys, values))
                    buffer_partd.append(data, lock=False)
                    logger.debug('Server appends %d keys', len(data))
                    self.ack(address)
                    self.nappended += 1
                    if self.nappend == self.nappended:
                        dprint(f'nappended reached nappend={self.nappend}')
                        self.close()

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

                else:
                    logger.debug("Unknown command: %s", command)
                    raise ValueError("Unknown command: " + str(command))

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
        print('server properly closed')
        self.status = 'closed'
        with suppress(zmq.error.ZMQError):
            self.socket.close(1)
        with suppress(zmq.error.ZMQError):
            self.context.destroy(3)


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
