

from ast import List
import os
from pathlib import Path
from time import gmtime, strftime
import time
import numpy as np
import pandas as pd
from typing import Union, List, Optional
from sklearn.linear_model import LinearRegression

def highest_value(input, current_max):
    if input > current_max:
        current_max = input
    # print(current_max)
    return current_max

# def record(signal_clean_TP9,signal_clean_AF7,signal_clean_AF8,signal_clean_TP10):
#     filename = os.path.join(os.getcwd(), ("C:/Users/drbar/OneDrive/Pulpit/Stage 2022/Recording/recording_%s.csv" % strftime("%Y-%m-%d-%H.%M.%S", gmtime())))

def record(data, timestamp, duration: int = 20,continuous: bool = True) -> None:

    filename = os.path.join(os.getcwd(), "C:/Users/drbar/OneDrive/Pulpit/Stage 2022/Recording/EEG_recording_%s.csv" %(strftime('%Y-%m-%d-%H.%M.%S', gmtime())))

    res = []
    timestamps = []
    t_init = time()
    last_written_timestamp = None
    print('Start recording at time t=%.3f' % t_init)
    # data, timestamp = inlet.pull_chunk(timeout=1.0, max_samples=chunk_length)

    res.append(data)
    timestamps.extend(timestamp)

    # Save every 5s
    if continuous and (last_written_timestamp is None or last_written_timestamp + 5 < timestamps[-1]):
        _save(
            filename,
            res,
            timestamps,
            last_written_timestamp=last_written_timestamp,
        )
        last_written_timestamp = timestamps[-1]

        

    _save(
        filename,
        res,
        timestamps,
    )

    print("Done - wrote file: {}".format(filename))


def _save(
    filename: Union[str, Path],
    res: list,
    timestamps: list,
    last_written_timestamp: Optional[float] = None,
):

    res = np.concatenate(res, axis=0)
    timestamps = np.array(timestamps)

    res = np.c_[timestamps, res]
    data = pd.DataFrame(data=res, columns=["timestamps","TP9","AF7","AF8","TP10"])

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # If file doesn't exist, create with headers
    # If it does exist, just append new rows
    if not Path(filename).exists():
        # print("Saving whole file")
        data.to_csv(filename, float_format='%.3f', index=False)
    else:
        # print("Appending file")
        # truncate already written timestamps
        data = data[data['timestamps'] > last_written_timestamp]
        data.to_csv(filename, float_format='%.3f', index=False, mode='a', header=False)