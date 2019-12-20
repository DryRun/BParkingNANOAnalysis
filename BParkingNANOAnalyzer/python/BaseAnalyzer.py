#! /usr/bin/env python

import ROOT
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
import time
import datetime
import uproot
import awkward
import sys
import os
import numpy as np
#import rootpy.ROOT as ROOT
from rootpy.io import root_open

#ROOT.gErrorIgnoreLevel=ROOT.kError

class BParkingNANOAnalyzer(object):
  def __init__(self, input_files, output_file, isMC):
    __metaclass__ = ABCMeta
    #self._file_out_name = outputfile.replace('.root','').replace('.h5','')
    self._input_files = input_files
    self._output_file = root_open(output_file, "RECREATE")
    self._isMC = isMC


  def print_timestamp(self):
    ts_start = time.time()
    print("[BParkingNANOAnalysis::print_timestamp] INFO : Time: {}".format(datetime.datetime.fromtimestamp(ts_start).strftime('%Y-%m-%d %H:%M:%S')))

  @abstractmethod
  def start(self):
    pass

  @abstractmethod
  def run(self):
    pass

  @abstractmethod
  def finish(self):
    pass








