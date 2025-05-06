"""
Author: Oliver Paull
Date: 11/08/2021

A few of the functions and classes here are derived from `refnx` classes for PLATYPUS neutron reflectometry, authored by Andrew Nelson. 
I have used these useful and generalised tools to apply this to TAIPAN.
"""

import io
import os
import os.path
import glob
import argparse
import re
import shutil
from time import gmtime, strftime
import string
import warnings
from contextlib import contextmanager
from operator import attrgetter
from numpy.lib.function_base import average
from scipy.optimize import leastsq, curve_fit
from scipy.stats import t
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py
import warnings
from lmfit import Minimizer, Parameter

def catalogue(start, stop, data_folder=None, prefix="TPN"):
    """
    Extract interesting information from Taipan NeXUS files.

    Parameters
    ----------
    start : int
        start cataloguing from this run number
    stop : int
        stop cataloguing at this run number
    data_folder : str, optional
        path specifying location of NeXUS files
    prefix : {'TPN'}, optional
        str specifying whether you want to catalogue Platypus or Spatz files

    Returns
    -------
    catalog : pd.DataFrame
        Dataframe containing interesting parameters from Platypus Nexus files
    """
    info = ["filename", "end_time", "sample_name"]



    info += [
        "end_time",
        "sample_description",
        "dE",
        "dqh",
        "dqk",
        "dql",
    ]

    run_number = []
    d = {key: [] for key in info}

    if data_folder is None:
        data_folder = "."

    files = glob.glob(os.path.join(data_folder, prefix + "*.nx.hdf"))
    files.sort()
    files = [
        file
        for file in files
        if datafile_number(file, prefix=prefix) in range(start, stop + 1)
    ]

    for idx, file in enumerate(files):
        pn = TaipanNexus(file)

        cat = pn.cat.cat
        cat["dE"] = max(cat["ei"]) - min(cat["ef"])
        cat["dqh"] = max(cat["qh"]) - min(cat["qh"])
        cat["dqk"] = max(cat["qk"]) - min(cat["qk"])
        cat["dql"] = max(cat["ql"]) - min(cat["ql"])

        run_number.append(idx)

        for key in d.keys():
            data = cat[key]
            if np.size(data) > 1 or type(data) is np.ndarray:
                data = data[0]
            if type(data) is bytes:
                data = data.decode()

            d[key].append(data)

    df = pd.DataFrame(d, index=run_number, columns=info)

    return df


class TaipanNexus(object):
    """
    Processes Taipan NeXus files to produce an intensity vs wavelength
    spectrum

    Parameters
    ----------
    h5data : HDF5 NeXus file or str
        An HDF5 NeXus file for Taipan, or a `str` containing the path
        to one
    """

    def __init__(self, h5data):
        """
        Initialises the TaipanNexus object.
        """
        self.prefix = "TPN"
        with _possibly_open_hdf_file(h5data, "r") as f:
            self.cat = TaipanCatalogue(f)
        
        self.average_qh = np.average(self.cat.qh)
        self.average_qk = np.average(self.cat.qk)
        self.average_ql = np.average(self.cat.ql)
        self.average_s1 = np.average(self.cat.s1)
        self.average_s2 = np.average(self.cat.s2)
        self.average_ei = np.average(self.cat.ei)
        self.average_ef = np.average(self.cat.ef)

        self.scan_axis = []

        for axis in ["qh", "qk", "ql", "s1", "s2", "ei", "ef", "sgu", "sgl"]:
            if average_neighbour_difference(getattr(self.cat, axis)) > 1e-4:
                self.scan_axis.append(axis)

        self.cts = self.cat.cat["bm2_counts"]
        self.cts_err = np.sqrt(self.cts)

    def __len__(self):
        return len(self.cts)

    def __str__(self):
        qh_min = min(self.cat.qh)
        qh_max = max(self.cat.qh)
        qk_min = min(self.cat.qk)
        qk_max = max(self.cat.qk)
        ql_min = min(self.cat.ql)
        ql_max = max(self.cat.ql)

        return (
            f"{len(self)} measured points in TPN file.\n"
            f"-------------------------------------\n"
            f"Q ranges:\nq_h \t ({qh_min:.3f}, {qh_max:.3f})\n"
            f"q_k \t ({qk_min:.3f}, {qk_max:.3f})\n"
            f"q_l \t ({ql_min:.3f}, {ql_max:.3f})"
        )

    def plot(self, axis=None, fig=None, ax=None, **kwargs):
        if axis is None:
            axis = self.scan_axis[0]

        fig, ax = plt.subplots()

        ax.plot(getattr(self.cat, axis), self.cts/self.cat.bm2_time, **kwargs)
        ax.set(
            xlabel=axis,
            ylabel="Counts/sec"
        )
        return fig, ax

    def save(
        self, 
        filename=None, 
        filetype="csv", 
        keys = ["qh", "qk", "ql", "bm2_counts"]
    ):
        d = {}
        for k in keys:
            d[k] = self.cat.cat[k]

        df = pd.DataFrame(d)

        if filename is None:
            filename = self.cat.filename.split()[0]
    
        if filetype == "csv":
            delimiter=","
        elif filetype == "txt":
            delimiter="\t"
        
        np.savetxt(f"{filename}.{filetype}", df, delimiter=delimiter)


class TaipanCatalogue(object):
    """
    Extract relevant parts of a TAIPAN NeXus file for analysis
    """

    def __init__(self, h5d):
        """
        Extract relevant parts of a NeXus file for reflectometry reduction
        Access information via dict access, e.g. cat['detector'].

        Parameters
        ----------
        h5d - HDF5 file handle
        """
        self.prefix = None

        d = {}
        file_path = os.path.realpath(h5d.filename)
        d["path"] = os.path.dirname(file_path)
        d["filename"] = os.path.basename(h5d.filename)
        try:
            d["end_time"] = h5d["entry1/end_time"][0]
        except KeyError:
            # Autoreduce field tests show that this key may not be present in
            # some files before final write.
            d["end_time"] = ""
        # Analyser angles 
        #d["a1"] = h5d["entry1/instrument/detector/a1"][:]
        d["a2"] = h5d["entry1/instrument/detector/a2"][:]
        # Effective omega and 2theta angles
        d["s1"] = h5d["entry1/sample/s1"][:]
        d["s2"] = h5d["entry1/sample/s2"][:]
        
        d["sgu"] = h5d["entry1/sample/sgu"][:]
        d["sgl"] = h5d["entry1/sample/sgl"][:]
        d["data"] = h5d["entry1/monitor/data"][:]
        d["bm1_time"] = h5d["entry1/monitor/bm1_time"][:]
        d["bm1_counts"] = h5d["entry1/monitor/bm1_counts"][:]
        d["bm2_time"] = h5d["entry1/monitor/bm2_time"][:]
        d["bm2_counts"] = h5d["entry1/monitor/bm2_counts"][:]
        
        d["sample_description"] = h5d["entry1/sample/description"][:]
        d["ef"] = h5d["entry1/sample/ef"][:]
        d["ei"] = h5d["entry1/sample/ei"][:]
        d["ki"] = h5d["entry1/sample/ki"][:]
        d["kf"] = h5d["entry1/sample/kf"][:]
        d["collimation"] = h5d["entry1/sample/name"][:]
        d["qh"] = h5d["entry1/sample/qh"][:]
        d["qk"] = h5d["entry1/sample/qk"][:]
        d["ql"] = h5d["entry1/sample/ql"][:]


        try:
            d["temp_1"] = h5d["entry1/sample/tc1/sensor/sensorValueA"][:]
            d["temp_1_setpoint1"] = h5d["entry1/sample/tc1/sensor/setpoint1"][:]
            d["temp_1_setpoint2"] = h5d["entry1/sample/tc1/sensor/setpoint2"][:]
        except KeyError:
            d["temp_1"] = None
            d["temp_1_setpoint1"] = None
            d["temp_1_setpoint2"] = None


        try:
            d["temp_2"] = h5d["entry1/sample/tc2/sensor/sensorValueA"][:]
            d["temp_2_setpoint1"] = h5d["entry1/sample/tc2/sensor/setpoint1"][:]
            d["temp_2_setpoint2"] = h5d["entry1/sample/tc2/sensor/setpoint2"][:]
        except KeyError:
            d["temp_2"] = None
            d["temp_2_setpoint1"] = None
            d["temp_2_setpoint2"] = None
        #d["temp"] = h5d["entry1/sample/tc1/sensor/sensorValueA"][:]

        
        try:
            d["start_time"] = h5d["entry1/instrument/detector/start_time"][:]
        except KeyError:
            # start times don't exist in this file
            d["start_time"] = None

        d["original_file_name"] = h5d["entry1/experiment/file_name"]
        d["sample_name"] = h5d["entry1/sample/name"][:]
        self.cat = d

    def __getattr__(self, item):
        return self.cat[item]

    @property
    def datafile_number(self):
        return datafile_number(self.filename, prefix=self.prefix)


class TaipanRSM(object):
    """
    Reciprocal space map measured by TAIPAN in a series of linescans. 

    Parameters
    ----------
    directory   :   str or os.path-like object
        directory to data files
    range       :   2-tuple
        (datafile_number_min, datafile_number_max)
        Min and max datafile numbers to be loaded for the reciprocal space map in ``directory``.
    verbose     :   bool
        Mainly for debugging and checking things
    """
    def __init__(self, directory, file_range=(0,1e10), verbose=False):
        self.scans = []
        self.step_axis = []


        for file in os.listdir(directory):
            if verbose:
                print(file)
            if not file.endswith(".nx.hdf"):
                continue
            if file_range[0] <= datafile_number(file) <= file_range[1]:
                self.scans.append(TaipanNexus(os.path.join(directory, file)))

        motor_averages = {}
        for scan in self.scans:
            for axis in ["qh", "qk", "ql", "s1", "s2", "ei", "ef"]:

                if axis not in scan.scan_axis:
                    motor_averages[axis] = getattr(scan, f"average_{axis}")

            
        for motor, diff in motor_averages.items():
            if diff > 1e-4:
                self.step_axis.append(motor)

        self.data = np.empty(
            [len(self.scans), len(self.scans[0].cts)],
            dtype=[("qh", "f4"), ("qk", "f4"), ("ql", "f4"), ("cts", "f4")]
        )

        self._load_files()
    
    def __len__(self):
        return len(self.scans)

    def __str__(self):
        qh_min = min(self.data["qh"].flatten())
        qh_max = max(self.data["qh"].flatten())
        qk_min = min(self.data["qk"].flatten())
        qk_max = max(self.data["qk"].flatten())
        ql_min = min(self.data["ql"].flatten())
        ql_max = max(self.data["ql"].flatten())

        return (
            f"{len(self)} TPN files in reciprocal space map.\n"
            f"-------------------------------------\n"
            f"Q ranges:\nq_h \t ({qh_min:.3f}, {qh_max:.3f})\n"
            f"q_k \t ({qk_min:.3f}, {qk_max:.3f})\n"
            f"q_l \t ({ql_min:.3f}, {ql_max:.3f})"
        )

    def sort_by(self, axis):
        """
        Sort files by attribute. Useful if your RSM files aren't measured in
        succession in reciprocal space.

        Parameters
        ----------
        axis    :   {"qh", "qk", "ql", "s1", "s2", "ei", "ef", "filename"}
        """

        if axis in ["qh", "qk", "ql", "s1", "s2", "ei", "ef"]:
            #rsm.sort(key=lambda x: np.average(getattr(x.cat, axis))
            scans = sorted(self.scans, key=attrgetter(f"average_{axis}"))
            self.scans = scans
        elif axis in "filename":
            scans = sorted([datafile_number(s.cat.filename) for s in self.scans])
            self.scans = scans

        self._load_files()

    def _load_files(self):
        for idx, scan in enumerate(self.scans):
            #print(idx)
            try:
                self.data["qh"][idx, :] = scan.cat.qh[:]
                self.data["qk"][idx, :] = scan.cat.qk[:]
                self.data["ql"][idx, :] = scan.cat.ql[:]
                self.data["cts"][idx, :] = scan.cts[:]
                #print(max(scan.cts))
            except ValueError:
                warnings.warn(
                    "Incomplete linescan detected. Attempting to fill in gaps by averaging."
                )
                # If one of the scans was aborted early for some reason, 
                # make a reasonable guess for what the rest of the motor positions will be, and fill out the missing arrays.
                normal_length = max([len(getattr(s, "cts")) for s in self.scans])
                num_missing = normal_length - len(scan.cts)
                qh_step = average_neighbour_difference(scan.cat.qh)
                qk_step = average_neighbour_difference(scan.cat.qk)
                ql_step = average_neighbour_difference(scan.cat.ql)
                
                # For missing detector data, get the average of the neighbouring linescans
                # and assign to missing point
                missing_counts = []
                for i in np.arange(len(scan.cat.qh),len(scan.cat.qh)+num_missing):
                    # If missing data is in the middle of good data, take average of neighbouring linescans
                    if (self.scans[idx] is not self.scans[-1]) and (self.scans[idx] is not self.scans[0]):
                        avg = np.average([
                            self.scans[idx-1].cts[i],
                            self.scans[idx+1].cts[i],
                        ])
                        missing_counts.append(avg)
                    # else the missing data is on the end-points of RSM, and we just take it as zero
                    else:
                        missing_counts.append(0)
                    
                counts = np.append(
                    scan.cts,
                    missing_counts
                )
                
                qh = np.append(
                    scan.cat.qh, 
                    np.linspace(
                        scan.cat.qh[-1]+qh_step,
                        scan.cat.qh[-1]+(num_missing)*qh_step,
                        num_missing
                    )
                )
                qk = np.append(
                    scan.cat.qk, 
                    np.linspace(
                        scan.cat.qk[-1]+qk_step,
                        scan.cat.qk[-1]+(num_missing)*qk_step,
                        num_missing
                    )
                )
                ql = np.append(
                    scan.cat.ql, 
                    np.linspace(
                        scan.cat.ql[-1]+ql_step,
                        scan.cat.ql[-1]+(num_missing)*ql_step,
                        num_missing
                    )
                )
                
                self.data["qh"][idx, :] = qh
                self.data["qk"][idx, :] = qk
                self.data["ql"][idx, :] = ql
                self.data["cts"][idx, :] = counts

    def plot(self, axis_1="qh", axis_2="ql", fig=None, ax=None, log=False, **kwargs):
        """
        Plot RSM quickly
        """
        fig, ax = plt.subplots()
        x = self.data[axis_1]
        y = self.data[axis_2]
        z = self.data["cts"]

        if log:
            z = np.log10(z)

        cs = ax.contourf(
            x,
            y,
            z,
            **kwargs
        )
        ax.set(
            xlabel=axis_1,
            ylabel=axis_2,
        )
        fig.colorbar(cs, label="Counts", ax=ax, shrink=0.9)

        return fig, ax


class Cycloid:
    """
    Describes and simulates the neutron scattering signal of an antiferromagnetic spin cycloid 
    with a given centre position and wavelength (TODO: add robust splitting direction input. Currently 
    assumes a splitting direction of [11-2]).

    Parameters
    ----------
    wavelength : float or `refnx.analysis.Parameter`
        Wavelength of the cycloid defining it's repeating unit 
        and corresponding splitting in reciprocal space
    position : tuple
        Tuple of the x and y centre position between the split cycloidal peaks
    params : dict
        dict of gaussian parameters to input to `vlabs.utils.Gauss2D`.
    """
    def __init__(self, wavelength, position, splitting=(-1,-1,2), params=None):
        

        self.x_centre = Parameter(
            value = position[0],
            name="XCEN",
            vary=False,
        )

        self.y_centre = Parameter(
            name="YCEN",
            value = position[1],
            vary=False,
        )

        if not isinstance(wavelength, Parameter):
            self.wavelength = Parameter(
                value=wavelength,
                name="WAVELENGTH",
                min=400,
                max=1500,
            )
        else:
            self.wavelength = wavelength

        self.splitting = splitting
        self.position = position

        self.calculate_cycloid()
        
        if params:
            if isinstance(params[0], Parameter):
                self.params = params
            elif isinstance(params[0], float):
                self.params["XCEN"] = Parameter(value=self.x_centre.value,name="XCEN", vary=False)
                self.params["YCEN"] = Parameter(value=self.y_centre.value,name="YCEN", vary=False)
                self.params["SIGMAX"] = Parameter(value=params["SIGMAX"], name="SIGMAX", min=1e-5, max=1, vary=True),
                self.params["SIGMAY"] = Parameter(value=params["SIGMAY"], name="SIGMAY", min=1e-5, max=1, vary=True),
                self.params["AMP"] = Parameter(value=params["AMP"], name="AMP", min=0, max=10000, vary=True),
                self.params["BACKGROUND"] = Parameter(value=params["BACKGROUND"], name="BACKGROUND", min=0, max=1000, vary=True),
                self.params["ANGLE"] = Parameter(value=params["ANGLE"],name="ANGLE", min=0, max=360, vary=True)
        else:
            self.params = {
                "XCEN" : Parameter(value=self.x_centre.value,name="XCEN", vary=False),
                "YCEN" : Parameter(value=self.y_centre.value,name="YCEN", vary=False),
                "SIGMAX" : Parameter(value=0.01, name="SIGMAX", min=1e-5, max=1, vary=True),
                "SIGMAY" : Parameter(value=0.01, name="SIGMAY", min=1e-5, max=1, vary=True),
                "AMP" : Parameter(value=400, name="AMP", min=0, max=10000, vary=True),
                "BACKGROUND" : Parameter(value=100, name="BACKGROUND", min=0, max=1000, vary=True),
                "ANGLE" : Parameter(value=0,name="ANGLE", min=0, max=360, vary=True)
            }
    
    def load_map(self, X, Y):
        """
        Load experimental qy and qz arrays to simulate data on

        Parameters
        ----------
        X : np.array 
            X values for reciprocal space map
        Y : np.array
            Y values for reciprocal space map
        """
        self.x_data = X
        self.y_data = Y
    
    def calculate_cycloid(self):
        q_space = 2*np.pi/self.wavelength.value
        # Get angle between cycloid splitting direction and [001]
        angle = angle_between(np.array(self.splitting), (0,0,1))
        split_x = q_space * np.cos(angle)
        split_y = q_space * np.sin(angle)
        
        self.delta_x = split_x/2
        self.delta_y = split_y/2

        self.satellite_1 = self.position + np.array([-self.delta_x, self.delta_y])
        self.satellite_2 = self.position + np.array([self.delta_x, -self.delta_y])
        return 

    def simulate_data(self):

        self.z_data = np.zeros(self.x_data.shape)
        for satellite in [self.satellite_1, self.satellite_2]:
            self.params["XCEN"].value = satellite[0]
            self.params["YCEN"].value = satellite[1]
            self.z_data += Gauss2d([self.x_data,self.y_data], **self.params)
            
        return self.z_data

    def simulate_datapoint(self, x, y):
        datapoint = np.zeros(x.shape)
        for satellite in [self.satellite_1, self.satellite_2]:
            self.params["XCEN"].value = satellite[0]
            self.params["YCEN"].value = satellite[1]
            datapoint += Gauss2d([x,y], **self.params)

        return datapoint



def average_neighbour_difference(array):
    """
    Gets the average step size for a given 1D array

    Parameters
    ----------
    array   :   (N,) np.array

    Returns
    ---------
    np.average(diff)    :   Average difference between 
                            neighbouring elements in the array 
    """
    if len(array) == 1:
        return array

    for i in range(len(array)-1):
        diff = array[i+1] - array[i]
    
    return np.average(diff)


def basename_datafile(pth):
    """
    Given a NeXUS path return the basename minus the file extension.
    Parameters
    ----------
    pth : str

    Returns
    -------
    basename : str

    Examples
    --------
    >>> basename_datafile('a/b/c.nx.hdf')
    'c'
    """

    basename = os.path.basename(pth)
    return basename.split(".nx.hdf")[0]


def number_datafile(run_number, prefix="TPN"):
    """
    Given a run number figure out what the file name is.
    Given a file name, return the filename with the .nx.hdf extension

    Parameters
    ----------
    run_number : int or str

    prefix : str, optional
        The instrument prefix. Only used if `run_number` is an int

    Returns
    -------
    file_name : str

    Examples
    --------
    >>> number_datafile(708)
    'PLP0000708.nx.hdf'
    >>> number_datafile(708, prefix='QKK')
    'QKK0000708.nx.hdf'
    >>> number_datafile('PLP0000708.nx.hdf')
    'PLP0000708.nx.hdf'
    """
    try:
        num = abs(int(run_number))
        # you got given a run number
        return "{0}{1:07d}.nx.hdf".format(prefix, num)
    except ValueError:
        # you may have been given full filename
        if run_number.endswith(".nx.hdf"):
            return run_number
        else:
            return run_number + ".nx.hdf"


def datafile_number(fname, prefix="TPN"):
    """
    From a filename figure out what the run number was

    Parameters
    ----------
    fname : str
        The filename to be processed

    Returns
    -------
    run_number : int
        The run number

    Examples
    --------
    >>> datafile_number('TPN0000708.nx.hdf')
    708

    """
    rstr = ".*" + prefix + "([0-9]{7}).nx.hdf"
    regex = re.compile(rstr)

    _fname = os.path.basename(fname)
    r = regex.search(_fname)

    if r:
        return int(r.groups()[0])

    return None

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    
    if np.linalg.norm(v1) == 0 or  np.linalg.norm(v2) == 0:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def Gauss2d(M, **params):
    """
    function to calculate any number of general two dimensional Gaussians. 
    Requires the x and y axes to be concatenated into a tuple of arrays, and
    that the number of parameters be divisible by the number of parameters
    for a 2D gaussian (i.e. 7) 

    Parameters
    ----------
    x, y :  array-like
        coordinate(s) where the function should be evaluated
    p :     list
        list of parameters of the Gauss-function
        [XCEN, YCEN, SIGMAX, SIGMAY, AMP, BACKGROUND, ANGLE];
        SIGMA = FWHM / (2*sqrt(2*log(2)));
        ANGLE = rotation of the X, Y direction of the Gaussian in radians

    Returns
    -------
    array-like
        the value of the Gaussian described by the parameters p at
        position (x, y)
    """
    x, y = M
    arr = np.zeros(x.shape)
    p = []
    if isinstance(params, dict):
        for key in params.keys():#["XCEN", "YCEN", "SIGMAX", "SIGMAY", "AMP", "BACKGROUND", "ANGLE"]:
            p.append(params[key])

@contextmanager
def _possibly_open_hdf_file(f, mode="r"):
    """
    Context manager for hdf5 files.

    Parameters
    ----------
    f : file-like or str
        If `f` is a file, then yield the file. If `f` is a str then open the
        file and yield the newly opened file.
        On leaving this context manager the file is closed, if it was opened
        by this context manager (i.e. `f` was a string).
    mode : str, optional
        mode is an optional string that specifies the mode in which the file
        is opened.

    Yields
    ------
    g : file-like
        On leaving the context manager the file is closed, if it was opened by
        this context manager.
    """
    close_file = False
    if type(f) == h5py.File:
        g = f
    else:
        g = h5py.File(f, mode)
        close_file = True
    yield g
    if close_file:
        g.close()
