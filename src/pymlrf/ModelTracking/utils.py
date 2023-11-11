import logging

logger = logging.getLogger("dtr_renal")

__all__ = ["Option"]

class Option:
    
    def __init__(self, option):
        if option not in ["overwrite", "duplicate", "error"]:
            raise TypeError("Values should only be one of overwrite, duplicate or error")
        self.option = option

    def __repr__(self):
        return str(self.option)