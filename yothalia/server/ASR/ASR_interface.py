from abc import ABC, abstractmethod

class ASR_interface(ABC):
    """A abstract class for all LLM
        must have an interface of input for 
        modularity and futuremaintainence
    """
    @abstractmethod
    def input(self, sentence) -> str:
        pass
    

    """@abstractmethod
    def output(self) -> tuple:
        pass"""




