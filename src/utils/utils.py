from typing import Callable
from utils.pylogger import get_pylogger

log = get_pylogger(__name__)


# utils/utils.py
# utils.py
import traceback

def task_wrapper(task_func):
    def wrap(*args, **kwargs):
        try:
            return task_func(*args, **kwargs)
        except Exception as ex:
            # Print detailed traceback to understand the root cause of the error
            print("An error occurred during the execution of the task:")
            traceback.print_exc()  # This prints the entire traceback of the exception
            raise ex  # Re-raise the exception to preserve the original traceback
    return wrap

# def task_wrapper(task_func):
#     def wrap(*args, **kwargs):
#         try:
#             return task_func(*args, **kwargs)
#         except Exception as ex:
#             print(f"Exception occurred during task execution:\n{ex}")
#             raise ex
#     return wrap
