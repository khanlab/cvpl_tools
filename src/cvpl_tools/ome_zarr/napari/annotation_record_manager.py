import os.path
import dask.array as da
import sqlite3
import numpy as np
import numpy.typing as npt
import pickle


class AnnotationRecordManager:
    """
    Manages useful info to be referenced in the annotation process
    Saves edit history for undo/redo support

    Manager states:
    - *annotate_edit table
    - *metadata table (pointers to annotate_edit)
    - *arrays
    - undo_stack (on the top are the latest undo actions)
    * means persisted above
    """

    def __init__(self, save_path: str):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        db_path = f'{save_path}/annotation_record.db'
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        self.masks = None
        self.mask_name_to_id = {}
        self.undo_stack = []  # empty if user has not done any undo; used for re-doing actions

        self.cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS metadata (
            id TEXT PRIMARY KEY,
            count INTEGER NOT NULL
        );
        ''')

        self.cursor.execute(f'''
        INSERT INTO metadata (id, count) VALUES ("num_sessions", 0) 
        ON CONFLICT(id) DO NOTHING;
        ''')

        self.cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS session (
            sid INTEGER PRIMARY KEY,
            stime DATE NOT NULL,
            etime DATE
        );
        ''')

        self.cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS annotation_edit (
            mid INTEGER,
            torder INTEGER,
            sid INTEGER REFERENCES session(sid) NOT NULL,
            position TEXT NOT NULL,
            size TEST NOT NULL,
            content_before TEXT NOT NULL,
            content_after TEXT NOT NULL,
            PRIMARY KEY (mid, torder)
        )
        ''')

        self.sid = self.inc_attr_count('num_sessions')

        self.cursor.execute('''
        INSERT INTO session VALUES (?, DATE('NOW'), ?)
        ON CONFLICT(sid) DO NOTHING;
        ''', (self.sid, None))  # when database connection closes, we will update the end time

    def get_sid(self) -> int:
        return self.sid

    def register_masks(self, masks: list[dict]):
        """Initialize the masks and the corresponding database information

        Not called on initialization because masks requires accessing database information to initialize
        In additional to necessary fields, masks dictionaries may have "on_update" and "on_close" callbacks
        defined.

        Args:
            masks: Dictionaries with fields "name", "arr", "path", "is_numpy" defined
        """
        self.masks = masks
        for i in range(len(masks)):
            mask_info = masks[i]
            for key in ("name", "arr", "path", "is_numpy"):
                assert key in mask_info, f'Key {key} missing in masks array entry {i}!'
        self.mask_name_to_id = {masks[i]['name']: i for i in range(len(masks))}
        mask_names = [mask_info['name'] for mask_info in masks]
        mask_init_values = ','.join(f'("{name}", 0)' for name in mask_names)

        self.cursor.execute(f'''
        INSERT INTO metadata (id, count) VALUES
        {mask_init_values}
        ON CONFLICT(id) DO NOTHING;
        ''')

        print('print database states on register:')
        self.print_table('metadata')
        self.print_table('session')

    def swap_region(self,
                    mask_id: npt.NDArray | da.Array,
                    stpt: npt.NDArray[np.int32],
                    edit_mask: npt.NDArray
                    ) -> npt.NDArray:
        mask = self.masks[mask_id]['arr']
        stpt = stpt.tolist()
        slices = tuple(slice(stpt[i], stpt[i] + edit_mask.shape[i]) for i in range(len(stpt)))
        swapped_region = mask[slices]
        mask[slices] = edit_mask
        if 'on_update' in self.masks[mask_id]:
            self.masks[mask_id]['on_update']()

        return swapped_region

    def do_edit_mask(self, mask_name: str, stpt: npt.NDArray[np.int32], edit_mask: npt.NDArray):
        self.clear_undo_stack()
        mask_id = self.mask_name_to_id[mask_name]

        edit_region_size = np.array(edit_mask.shape, dtype=np.int32)
        swapped_region = self.swap_region(mask_id, stpt, edit_mask)
        order = self.inc_attr_count(mask_name, decrease=False)

        self.insert_row(mask_id, order, stpt, edit_region_size, swapped_region, edit_mask)

    def undo_edit_mask(self, mask_name: str):
        mask_id = self.mask_name_to_id[mask_name]
        order = self.get_attr_count(mask_name)
        if order == 0:
            return

        result = self.pop_row(mask_id, order, delete=False)
        _, _, _, position, _, content_before, _ = result
        self.undo_stack.append((mask_id, order))
        self.swap_region(mask_id, position, content_before)

        self.inc_attr_count(mask_name, decrease=True)

    def redo_edit_mask(self):
        if len(self.undo_stack) == 0:
            return
        mask_id, order = self.undo_stack.pop()
        mask_name = self.masks[mask_id]['name']
        result = self.pop_row(mask_id, order, delete=False)
        _, _, _, position, _, _, content_after = result
        self.swap_region(mask_id, position, content_after)

        self.inc_attr_count(mask_name, decrease=False)

    def clear_undo_stack(self):
        while len(self.undo_stack) > 0:
            mask_id, order = self.undo_stack.pop()
            self.pop_row(mask_id, order, delete=True)

    def insert_row(self,
                   mask_id: int,
                   order: int,
                   stpt: npt.NDArray[np.int32],
                   edit_region_size: npt.NDArray[np.int32],
                   content_before: npt.NDArray,
                   content_after: npt.NDArray):
        self.cursor.execute('''
                        INSERT INTO annotation_edit VALUES
                        (?, ?, ?, ?, ?, ?, ?)
                        ''', (
            mask_id,
            order,
            self.sid,
            pickle.dumps(stpt),
            pickle.dumps(edit_region_size),
            pickle.dumps(content_before),
            pickle.dumps(content_after)
        ))

    def pop_row(self, mask_id: int, order: int, delete: bool = True) -> tuple:
        self.cursor.execute('''
        SELECT * FROM annotate_edit
        WHERE mid = ? AND torder = ?;
        ''', (mask_id, order))

        result = self.cursor.fetchone()
        if delete:
            self.cursor.execute('''
            DELETE FROM annotate_edit WHERE mid = ? AND torder = ?
            ''', (mask_id, order))
        for idx in (3, 4, 5, 6):
            result[idx] = pickle.loads(result[idx])  # these are numpy arrays

        return result

    def get_attr_count(self, attr_name: str) -> int:
        self.cursor.execute('''
        SELECT count FROM metadata WHERE id = ?
        ''', (attr_name,))
        result = self.cursor.fetchone()[0]
        assert isinstance(result, int), f'Expected int, got {type(result)}'
        return result

    def inc_attr_count(self, attr_name: str, decrease: bool = False) -> int:
        self.cursor.execute('''
        SELECT count FROM metadata WHERE id = ?
        ''', (attr_name,))
        result = self.cursor.fetchone()[0]
        assert isinstance(result, int), f'Expected int, got {type(result)}'
        if decrease:
            result -= 1
        else:
            result += 1
        self.cursor.execute('''
        UPDATE metadata SET count = ? WHERE id = ?;
        ''', (result, attr_name))
        return result

    def print_table(self, tname: str):
        print('-' * 30)
        print(f'{tname}')
        self.cursor.execute(f"SELECT * FROM {tname}")
        print(self.cursor.fetchall())

    def commit(self):
        self.clear_undo_stack()
        self.conn.commit()

    def close(self):
        self.cursor.execute('''
        UPDATE session
        SET etime = DATE('NOW')
        WHERE sid = ?
        ''', (self.sid,))  # when database connection closes, we will update the end time
        self.clear_undo_stack()  # commit is done here

        for mask_id in range(len(self.masks)):
            if 'on_close' in self.masks[mask_id]:
                self.masks[mask_id]['on_close']()
        self.cursor.close()
        self.conn.close()

