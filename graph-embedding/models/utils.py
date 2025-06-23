import hashlib
import json
from uuid import UUID, uuid5

from database.database import DatabaseWrapper


def data_to_database(data):
    database = DatabaseWrapper()
    return database.generate_train_srldatabase(data, use_recursion=True)


def set_to_uuid(s):
    namespace = UUID('00000000-0000-0000-0000-000000000000')
    key_base = json.dumps(sorted(s), sort_keys=True).encode('utf-8')
    return uuid5(namespace, hashlib.sha1(key_base).hexdigest())