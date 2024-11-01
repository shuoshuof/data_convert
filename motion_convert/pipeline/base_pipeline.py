import os
import pickle
from tqdm import tqdm
import torch
from abc import ABC, abstractmethod
from typing import Optional, Union
from typing import TypedDict
import multiprocessing
import numpy as np
from tqdm import tqdm
import pandas as pd
from typing_extensions import NotRequired

class PipelineArgs(TypedDict):
    max_epoch:NotRequired[int]
    lr:NotRequired[float]
    clip_angle: NotRequired[bool]

    generate_mirror:NotRequired[bool]

    zero_root:NotRequired[bool]
    filter:NotRequired[bool]
    fix_joints:NotRequired[bool]
    joint_indices:NotRequired[list]
    fix_ankles:NotRequired[bool]
    height_adjustment:NotRequired[bool]
    move_to_ground:NotRequired[bool]

    save_info: bool



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
    def _process_data(self,data_chunk,results,process_idx,debug,**kwargs):
        pass
    def _update_pbar(self,results,num_data):
        with tqdm(total=num_data) as pbar:
            while True:
                percent = len(results)
                pbar.n = percent
                pbar.refresh()
                if len(results) == num_data:
                    break

    def run(self,debug=False,**kwargs:PipelineArgs):
        data = self._read_data(**kwargs)

        data_chunks = self._split_data(data,**kwargs)

        manager = multiprocessing.Manager()
        results = manager.list([])

        os.makedirs(self.save_dir,exist_ok=True)

        if debug:
            self._process_data(data_chunks[0], results, 0,debug,**kwargs)
        else:
            processes = []
            for process_idx in range(self.num_processes):
                p = multiprocessing.Process(target=self._process_data,
                                            args=(data_chunks[process_idx], results, process_idx,debug),
                                            kwargs=kwargs)
                processes.append(p)
                p.start()
            # pbar process
            pbar_process = multiprocessing.Process(target=self._update_pbar,args=(results,len(data)))
            processes.append(pbar_process)
            pbar_process.start()

            for p in processes:
                p.join()

        results = pd.DataFrame(list(results))
        print(results)
        if kwargs.get('save_info',False):
            results.to_csv(os.path.join(self.save_dir,'results.csv'),index=False)
        print("all processes done")

        return results
