from dataclasses import dataclass

__all__ = ["Option"]

@dataclass
class Option:
    
    option:str
    __valid_options = ["overwrite", "duplicate", "error", "update"]
    
    def __post_init__(self):
        if self.option not in self.__valid_options:
            raise TypeError(
                f"Values should only be one of {self.__valid_options}"
                )

    def __repr__(self):
        return str(self.option)