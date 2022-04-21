from abc import ABCMeta, abstractmethod
import numpy as np
from math import log2
import cmath

h_bar = 1 # 1.054571817*10**(-34) # in J.s

class Model(metaclass=ABCMeta):
        """Interface class """

        def __init__(self, name, model_hamiltonian):
                self.__callables = ['load_settings']
                self.__attributes = self.__callables + []
                self.name = name
                self.model_hamiltonian = model_hamiltonian
                self.eig_values = None
                self.eig_vectors = None
                self.ground_states = None

        @property
        def attributes(self):
                """returns a list of attributes that must be defined"""
                return self.__attributes

        @property
        def callables(self):
                """returns a list of callables that must be defined"""
                return self.__callables

        def __subclasscheck__(self, other: object):
                if all(hasattr(other, attr) for attr in self.__attributes):
                        if all(callable(getattr(other, attr)) for attr in self.__callables):
                                return True
                        return False

        def __check_reason__(self):
            return [all(hasattr(self, attr) for attr in self.__attributes), all(callable(getattr(self, attr)) for attr in self.__callables)]

        def __check_validity__(self):
            """ Asserts that all necassery attributes are defined and all expected methods callable"""
            __reasons = self.__check_reason__()
            assert all(__reasons), "The specialization {} is not a proper subclass of {}!\nAll attributes defined {}, all callables defined {}".format(self.__name__,self.__bases__[0],__reasons[0],__reasons[1])

        @abstractmethod
        def load_settings(self, settings):
                """place holder"""
                raise NotImplementedError

        def parametrize_hamiltonian(self,*parameters):
                
                
