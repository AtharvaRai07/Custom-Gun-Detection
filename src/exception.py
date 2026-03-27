import sys
from src.logger import logging

class CustomException(Exception):
    "Base class for custom exceptions in the application."
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)

        self.error_message = error_message
        _, _, exc_tb = error_detail.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename

    def str(self):
        return f"Error occurred in script: {self.filename} at line number: {self.lineno} with message: {self.error_message}"
