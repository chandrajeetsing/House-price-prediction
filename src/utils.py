# src/utils.py

import os
import dill
import numpy as np

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            # We use 'dill' which is often better than 'pickle' for ML objects
            dill.dump(obj, file_obj)

    except Exception as e:
        raise e