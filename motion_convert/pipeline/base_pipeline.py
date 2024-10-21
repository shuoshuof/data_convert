import os

from tqdm import tqdm
import torch
from abc import ABC, abstractmethod
from typing import Optional, Union
import multiprocessing
import numpy as np

class BasePipeline(ABC):
    def __init__(self,motion_dir:str,save_dir:str,num_processes:int=None):
        self.num_processes = multiprocessing.cpu_count() if num_processes is None else num_processes
        self.motion_dir = motion_dir
        self.save_dir = save_dir
    @abstractmethod
    def _read_data(self,**kwargs)->Optional[list]:
        pass
    @abstractmethod
    def _split_data(self,data,**kwargs)->Union[list,np.ndarray]:
        pass
    @abstractmethod
    def _process_data(self,data_chunk,results,process_idx,**kwargs):
        pass
    def run(self,debug=False,**kwargs):
        data = self._read_data(**kwargs)

        data_chunks = self._split_data(data,**kwargs)

        manager = multiprocessing.Manager()
        results = manager.list([None]*self.num_processes)

        os.makedirs(self.save_dir,exist_ok=True)

        if debug:
            self._process_data(data_chunks[0], results, 0,**kwargs)
        else:
            processes = []
            for process_idx in range(self.num_processes):
                p = multiprocessing.Process(target=self._process_data,
                                            args=(data_chunks[process_idx], results, process_idx),
                                            kwargs=kwargs)
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

        print("all processes done")



