#! /usr/bin/env python

#import ROOT
import sys
import awkward
import uproot
import uproot_methods
import pandas as pd
import numpy as np
import math
from rootpy.plotting import Hist
from rootpy.io import root_open
from rootpy.plotting import Hist
from rootpy.tree import Tree
from root_numpy import fill_hist, array2root, array2tree
import root_pandas
import copy

from BParkingNANOAnalysis.BParkingNANOAnalyzer.BaseAnalyzer import BParkingNANOAnalyzer
from BParkingNANOAnalysis.BParkingNANOAnalyzer.constants import *

np.set_printoptions(threshold=np.inf)

def where(predicate, iftrue, iffalse):
    predicate = predicate.astype(numpy.bool)   # just to make sure they're 0/1
    return predicate*iftrue + (1 - predicate)*iffalse

class Bu2KJpsi2KMuMuAnalyzer(BParkingNANOAnalyzer):
  def __init__(self, input_files, output_file, isMC=False):
    super(Bu2KJpsi2KMuMuAnalyzer, self).__init__(input_files, output_file, isMC)
    self._trigger = "HLT_Mu7_IP4"

  def start(self):
    self._bu_branchnames = [
      "BToKMuMu_chi2",
      "BToKMuMu_cos2D",
      "BToKMuMu_eta",
      "BToKMuMu_fit_cos2D",
      "BToKMuMu_fit_eta",
      "BToKMuMu_fit_k_eta",
      "BToKMuMu_fit_k_phi",
      "BToKMuMu_fit_k_pt",
      "BToKMuMu_fit_l1_eta",
      "BToKMuMu_fit_l1_phi",
      "BToKMuMu_fit_l1_pt",
      "BToKMuMu_fit_l2_eta",
      "BToKMuMu_fit_l2_phi",
      "BToKMuMu_fit_l2_pt",
      "BToKMuMu_fit_mass",
      "BToKMuMu_fit_massErr",
      "BToKMuMu_fit_phi",
      "BToKMuMu_fit_pt",
      "BToKMuMu_l_xy",
      "BToKMuMu_l_xy_unc",
      "BToKMuMu_mass",
      "BToKMuMu_maxDR",
      "BToKMuMu_minDR",
      "BToKMuMu_mllErr_llfit",
      "BToKMuMu_mll_fullfit",
      "BToKMuMu_mll_llfit",
      "BToKMuMu_mll_raw",
      "BToKMuMu_phi",
      "BToKMuMu_pt",
      "BToKMuMu_svprob",
      "BToKMuMu_vtx_ex",
      "BToKMuMu_vtx_ey",
      "BToKMuMu_vtx_ez",
      "BToKMuMu_vtx_x",
      "BToKMuMu_vtx_y",
      "BToKMuMu_vtx_z",
      "BToKMuMu_charge",
      "BToKMuMu_kIdx",
      "BToKMuMu_l1Idx",
      "BToKMuMu_l2Idx",
      "BToKMuMu_pdgId",
    ]
    self._event_branchnames = [
      "nMuon",
      "nBToKMuMu",
      "HLT_Mu7_IP4",
      "HLT_Mu8_IP6",
      "HLT_Mu8_IP5",
      "HLT_Mu8_IP3",
      "HLT_Mu8p5_IP3p5",
      "HLT_Mu9_IP6",
      "HLT_Mu9_IP5",
      "HLT_Mu9_IP4",
      "HLT_Mu10p5_IP3p5",
      "HLT_Mu12_IP6",
      "L1_SingleMu7er1p5",
      "L1_SingleMu8er1p5",
      "L1_SingleMu9er1p5",
      "L1_SingleMu10er1p5",
      "L1_SingleMu12er1p5",
      "L1_SingleMu22",
      "nProbeTracks",      
      "nTrigObj",
      "nTriggerMuon",
      "event",
    ]
    self._muon_branchnames = [
      "Muon_dxy",
      "Muon_dxyErr",
      "Muon_dz",
      "Muon_dzErr",
      "Muon_eta",
      "Muon_ip3d",
      "Muon_mass",
      "Muon_pfRelIso03_all",
      "Muon_pfRelIso03_chg",
      "Muon_pfRelIso04_all",
      "Muon_phi",
      "Muon_pt",
      "Muon_ptErr",
      "Muon_segmentComp",
      "Muon_sip3d",
      "Muon_vx",
      "Muon_vy",
      "Muon_vz",
      "Muon_charge",
      "Muon_isTriggering",
      "Muon_nStations",
      "Muon_pdgId",
      "Muon_tightCharge",
      "Muon_highPtId",
      "Muon_inTimeMuon",
      "Muon_isGlobal",
      "Muon_isPFcand",
      "Muon_isTracker",
      "Muon_mediumId",
      "Muon_mediumPromptId",
      "Muon_miniIsoId",
      "Muon_multiIsoId",
      "Muon_mvaId",
      "Muon_pfIsoId",
      "Muon_softId",
      "Muon_softMvaId",
      "Muon_tightId",
      "Muon_tkIsoId",
      "Muon_triggerIdLoose",
    ]
    self._track_branchnames = [ 
      "ProbeTracks_DCASig",
      "ProbeTracks_dxy",
      "ProbeTracks_dxyS",
      "ProbeTracks_dz",
      "ProbeTracks_dzS",
      "ProbeTracks_eta",
      "ProbeTracks_mass",
      "ProbeTracks_phi",
      "ProbeTracks_pt",
      "ProbeTracks_vx",
      "ProbeTracks_vy",
      "ProbeTracks_vz",
      "ProbeTracks_charge",
      "ProbeTracks_isLostTrk",
      "ProbeTracks_isPacked",
      "ProbeTracks_pdgId",
      "ProbeTracks_isMatchedToEle",
      "ProbeTracks_isMatchedToLooseMuon",
      "ProbeTracks_isMatchedToMediumMuon",
      "ProbeTracks_isMatchedToMuon",
      "ProbeTracks_isMatchedToSoftMuon",
    ]
    self._trigobj_branchnames = [
      "TrigObj_pt",
      "TrigObj_eta",
      "TrigObj_phi",
      "TrigObj_l1pt",
      "TrigObj_l1pt_2",
      "TrigObj_l2pt",
      "TrigObj_id",
      "TrigObj_l1iso",
      "TrigObj_l1charge",
      "TrigObj_filterBits",
    ]
    self._muon_trigmatched_branchnames = [
      "TriggerMuon_eta",
      "TriggerMuon_mass",
      "TriggerMuon_phi",
      "TriggerMuon_pt",
      "TriggerMuon_vx",
      "TriggerMuon_vy",
      "TriggerMuon_vz",
      "TriggerMuon_charge",
      "TriggerMuon_pdgId",
      "TriggerMuon_trgMuonIndex",
    ]
    self._gen_branchnames = [
      "nGenPart",
      "GenPart_eta",
      "GenPart_mass",
      "GenPart_phi",
      "GenPart_pt",
      "GenPart_vx",
      "GenPart_vy",
      "GenPart_vz",
      "GenPart_genPartIdxMother",
      "GenPart_pdgId",
      "GenPart_status",
      "GenPart_statusFlags",
    ]

    #if self._isMC:
    #  self._input_branches.extend(['GenPart_pdgId', 'GenPart_genPartIdxMother'])

    self._mu_histograms = {}
    self._mu_histograms["nMuon"]          = Hist(11,-0.5, 10.5, name="nMuon", title="", type="F")
    self._mu_histograms["nMuon_isTrig"]   = Hist(11,-0.5, 10.5, name="nMuon_isTrig", title="", type="F")
    self._mu_histograms["Muon_pt"]        = Hist(100, 0.0, 100.0, name="Muon_pt", title="", type="F")
    self._mu_histograms["Muon_pt_isTrig"] = Hist(100, 0.0, 100.0, name="Muon_pt_isTrig", title="", type="F")

    self._histograms = {}
    self._histograms = {}
    for tag_type in ["inclusive", "triggered", "tag", "probe"]:
      self._histograms[tag_type] = {}
      self._histograms[tag_type] = {}

      self._histograms[tag_type]["BToKMuMu_chi2"]            = Hist(100, 0.0, 100.0, name="{}_BToKMuMu_chi2".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_eta"]             = Hist(50, -5.0, 5.0, name="{}_BToKMuMu_eta".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_fit_cos2D"]       = Hist(100, -1., 1., name="{}_BToKMuMu_fit_cos2D".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_fit_eta"]         = Hist(50, -5.0, 5.0, name="{}_BToKMuMu_fit_eta".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_fit_mass"]        = Hist(100, BU_MASS * 0.9, BU_MASS * 1.1, name="{}_BToKMuMu_fit_mass".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_fit_phi"]         = Hist(50, -2.0*math.pi, 2.0*math.pi, name="{}_BToKMuMu_fit_phi".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_fit_pt"]          = Hist(100, 0.0, 100.0, name="{}_BToKMuMu_fit_pt".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_l_xy"]            = Hist(50, -1.0, 4.0, name="{}_BToKMuMu_l_xy".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_l_xy_sig"]        = Hist(50, -1.0, 4.0, name="{}_BToKMuMu_l_xy_sig".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_fit_l1_eta"]      = Hist(50, -5.0, 5.0, name="{}_BToKMuMu_fit_l1_eta".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_fit_l1_phi"]      = Hist(50, -2.0*math.pi, 2.0*math.pi, name="{}_BToKMuMu_fit_l1_phi".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_fit_l1_pt"]       = Hist(100, 0.0, 100.0, name="{}_BToKMuMu_fit_l1_pt".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_fit_l2_eta"]      = Hist(50, -5.0, 5.0, name="{}_BToKMuMu_fit_l2_eta".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_fit_l2_phi"]      = Hist(50, -2.0*math.pi, 2.0*math.pi, name="{}_BToKMuMu_fit_l2_phi".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_fit_l2_pt"]       = Hist(100, 0.0, 100.0, name="{}_BToKMuMu_fit_l2_pt".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_mass"]            = Hist(500, BU_MASS * 0.9, BU_MASS * 1.1, name="{}_BToKMuMu_mass".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_mll_fullfit"]     = Hist(500, JPSI_1S_MASS * 0.9, JPSI_1S_MASS * 1.1, name="{}_BToKMuMu_mll_fullfit".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_mll_llfit"]       = Hist(500, JPSI_1S_MASS * 0.9, JPSI_1S_MASS * 1.1, name="{}_BToKMuMu_mll_llfit".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_mll_raw"]         = Hist(500, JPSI_1S_MASS * 0.9, JPSI_1S_MASS * 1.1, name="{}_BToKMuMu_mll_raw".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_phi"]             = Hist(50, -2.0*math.pi, 2.0*math.pi, name="{}_BToKMuMu_phi".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_pt"]              = Hist(100, 0.0, 100.0, name="{}_BToKMuMu_pt".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_svprob"]          = Hist(100, -1.0, 1.0, name="{}_BToKMuMu_svprob".format(tag_type), title="", type="F")
      self._histograms[tag_type]["BToKMuMu_charge"]          = Hist(3, -1.5, 1.5, name="{}_BToKMuMu_charge".format(tag_type), title="", type="F")

      self._histograms[tag_type]["BToKMuMu_fit_l_minpt"]      = Hist(100, 0.0, 100.0, name="{}_BToKMuMu_fit_l_minpt".format(tag_type), title="", type="F")


    self._cutflow_names = []
    self._cutflow_names.append("Inclusive")
    self._cutflow_names.append(self._trigger)
    self._cutflow_names.append("Inclusive SV")
    self._cutflow_names.append("Inclusive mu-K")
    self._cutflow_names.append("Inclusive Jpsi")

    self._cutflow_names.append("Tag")
    self._cutflow_names.append("Tag SV")
    self._cutflow_names.append("Tag mu-K")
    self._cutflow_names.append("Tag Jpsi")

    self._cutflow_names.append("Probe")
    self._cutflow_names.append("Probe SV")
    self._cutflow_names.append("Probe mu-K")
    self._cutflow_names.append("Probe Jpsi")

    self._cutflow_counts = {}
    for name in self._cutflow_names:
      self._cutflow_counts[name] = 0

  def run(self):
    print('[Bu2KJpsi2KMuMuAnalyzer::run] INFO: Running the analyzer...')
    self.print_timestamp()
    for ifile, filename in enumerate(self._input_files):
      print('[Bu2KJpsi2KMuMuAnalyzer::run] INFO: FILE: {}/{}. Getting branches from file...'.format(ifile, len(self._input_files)))
      tree = uproot.open(filename)['Events']
      self._bu_branches = {key: awkward.fromiter(branch) for key, branch in tree.arrays(self._bu_branchnames).items()}
      self._event_branches = {key: awkward.fromiter(branch) for key, branch in tree.arrays(self._event_branchnames).items()}
      self._muon_branches = {key: awkward.fromiter(branch) for key, branch in tree.arrays(self._muon_branchnames).items()}
      self._track_branches = {key: awkward.fromiter(branch) for key, branch in tree.arrays(self._track_branchnames).items()}
      self._gen_branches = {key: awkward.fromiter(branch) for key, branch in tree.arrays(self._gen_branchnames).items()}

      print('[Bu2KJpsi2KMuMuAnalyzer::run] INFO: FILE: {}/{}. Analyzing...'.format(ifile, len(self._input_files)))

      # Muon information
      self._muon_branches["Muon_isTriggeringBool"] = (self._muon_branches["Muon_isTriggering"] == 1)
      fill_hist(self._mu_histograms["nMuon"], self._muon_branches["Muon_pt"].count())
      fill_hist(self._mu_histograms["nMuon_isTrig"], self._muon_branches["Muon_pt"][self._muon_branches["Muon_isTriggeringBool"]].count())
      fill_hist(self._mu_histograms["Muon_pt"], self._muon_branches["Muon_pt"].flatten())
      fill_hist(self._mu_histograms["Muon_pt_isTrig"], self._muon_branches["Muon_pt"][self._muon_branches["Muon_isTriggeringBool"]].flatten())

      # Tag/probe determination
      isTrig_mu1 = self._muon_branches["Muon_isTriggering"][self._bu_branches["BToKMuMu_l1Idx"]] # shape=BToKMuMu
      isTrig_mu2 = self._muon_branches["Muon_isTriggering"][self._bu_branches["BToKMuMu_l2Idx"]] # shape=BToKMuMu
      bu_trig_count = isTrig_mu1 + isTrig_mu2 # shape=BToKMuMu
      total_trig_count = self._muon_branches["Muon_isTriggering"].sum() # shape=Event simple array
      total_trig_count_bushape = bu_trig_count.ones_like() * total_trig_count
      tag_count = total_trig_count_bushape - bu_trig_count
      self._bu_branches["BToKMuMu_isTag"] = (isTrig_mu1 == 1) | (isTrig_mu2 == 1)
      self._bu_branches["BToKMuMu_isProbe"] = (tag_count >= 1)

      if ifile == 0:
        print("Muon debug info:")
        print(self._event_branches["nMuon"])
        print(self._muon_branches["Muon_pt"].count()[:6])
        print(self._muon_branches["Muon_pt"][:6])
        #print(self._muon_branches["Muon_isTriggering"])
        print(self._muon_branches["Muon_isTriggeringBool"][:6])
        print (self._muon_branches["Muon_pt"][self._muon_branches["Muon_isTriggeringBool"]][:6])

        print("BToKMuMu_l1Idx = ")
        print(self._bu_branches["BToKMuMu_l1Idx"][:6])
        print("BToKMuMu_l2Idx = ")
        print(self._bu_branches["BToKMuMu_l2Idx"][:6])
        print("Total_trig_count = ")
        print(self._muon_branches["Muon_isTriggering"].sum()[:6])
        print("isTrig_mu1 = ")
        print(self._muon_branches["Muon_isTriggering"][self._bu_branches["BToKMuMu_l1Idx"]][:6])
        print("isTrig_mu2 = ")
        print(self._muon_branches["Muon_isTriggering"][self._bu_branches["BToKMuMu_l2Idx"]][:6])

        print("bu_trig_count = ")
        print(bu_trig_count[:6])
        print("total_trig_count_bushape = ")
        print(total_trig_count_bushape[:6])
        print("tag_count = ")
        print(tag_count[:6])
        print "isTag:"
        print(self._bu_branches["BToKMuMu_isTag"][:6])
        print "isProbe:"
        print(self._bu_branches["BToKMuMu_isProbe"][:6])

      # MC truth matching
      self._bu_branches["BToKMuMu_l1_genIdx"] = self._muon_branches["Muon_genPartIdx"][self._bu_branches["BToKMuMu_l1Idx"]] 
      self._bu_branches["BToKMuMu_l2_genIdx"] = self._muon_branches["Muon_genPartIdx"][self._bu_branches["BToKMuMu_l2Idx"]] 
      self._bu_branches['BToKMuMu_k_genIdx']  = self._track_branches['ProbeTracks_genPartIdx'][self._branches['BToKMuMu_kIdx']]

      self._bu_branches['BToKMuMu_l1_genMotherIdx'] = where(self._bu_branches["BToKMuMu_l1_genIdx"] >= 0, 
                                                                self._gen_branches["GenPart_genPartIdxMother"][self._bu_branches["BToKMuMu_l1_genIdx"]], 
                                                                -1)
      self._bu_branches['BToKMuMu_l2_genMotherIdx'] = where(self._bu_branches["BToKMuMu_l2_genIdx"] >= 0, 
                                                                self._gen_branches["GenPart_genPartIdxMother"][self._bu_branches["BToKMuMu_l2_genIdx"]], 
                                                                -1)
      self._bu_branches['BToKMuMu_k_genMotherIdx'] = where(self._bu_branches["BToKMuMu_k_genIdx"] >= 0, 
                                                                self._gen_branches["GenPart_genPartIdxMother"][self._bu_branches["BToKMuMu_k_genIdx"]], 
                                                                -1)

      self._bu_branches['BToKMuMu_l1_genGrandmotherIdx'] = where(self._bu_branches['BToKMuMu_l1_genMotherIdx'] >= 0, 
                                                                self._gen_branches["GenPart_genPartIdxMother"][self._bu_branches['BToKMuMu_l1_genMotherIdx']], 
                                                                -1)
      self._bu_branches['BToKMuMu_l2_genGrandmotherIdx'] = where(self._bu_branches['BToKMuMu_l2_genMotherIdx'] >= 0, 
                                                                self._gen_branches["GenPart_genPartIdxMother"][self._bu_branches['BToKMuMu_l2_genMotherIdx']], 
                                                                -1)

      self._bu_branches['BToKMuMu_l1_genMotherPdgId'] = where(self._bu_branches['BToKMuMu_l1_genMotherIdx'] >= 0, 
                                                                self._gen_branches["GenPart_pdgId"][self._bu_branches['BToKMuMu_l1_genMotherIdx']],
                                                                -1)
      self._bu_branches['BToKMuMu_l2_genMotherPdgId'] = where(self._bu_branches['BToKMuMu_l2_genMotherIdx'] >= 0, 
                                                                self._gen_branches["GenPart_pdgId"][self._bu_branches['BToKMuMu_l2_genMotherIdx']],
                                                                -1)
      self._bu_branches['BToKMuMu_k_genMotherPdgId'] = where(self._bu_branches['BToKMuMu_k_genMotherIdx'] >= 0, 
                                                                self._gen_branches["GenPart_pdgId"][self._bu_branches['BToKMuMu_k_genMotherIdx']],
                                                                -1)

      self._bu_branches['BToKMuMu_l1_genGrandmotherPdgId'] = where(self._bu_branches['BToKMuMu_l1_genGrandmotherIdx'] >= 0, 
                                                                self._gen_branches["GenPart_pdgId"][self._bu_branches['BToKMuMu_l1_genGrandmotherIdx']],
                                                                -1)
      self._bu_branches['BToKMuMu_l2_genGrandmotherPdgId'] = where(self._bu_branches['BToKMuMu_l2_genGrandmotherIdx'] >= 0, 
                                                                self._gen_branches["GenPart_pdgId"][self._bu_branches['BToKMuMu_l2_genGrandmotherIdx']],
                                                                -1)

      self._bu_branches['BToKMuMu_mcmatch'] = (self._bu_branches['BToKMuMu_l1_genMotherPdgId'] == 443) 
                                              & (self._bu_branches['BToKMuMu_l2_genMotherPdgId'] == 443) 
                                              & (self._bu_branches['BToKMuMu_l2_genGrandmotherPdgId'] == 521) 
                                              & (self._bu_branches['BToKMuMu_l2_genGrandmotherPdgId'] == 521) 
                                              & (self._bu_branches['BToKMuMu_k_genMotherPdgId'] == 521) 
                                              & (self._bu_branches['BToKMuMu_l1_genGrandmotherIdx'] == self._bu_branches['BToKMuMu_l2_genGrandmotherIdx']) 
                                              & (self._bu_branches['BToKMuMu_l1_genGrandmotherIdx'] == self._bu_branches['BToKMuMu_k_genGrandmotherIdx']) 

      self._bu_branches["BToKMuMu_genPartIdx"] = where(self._bu_branches['BToKMuMu_mcmatch'], self._bu_branches['BToKMuMu_l1_genGrandmotherIdx'], -1)

      self._butruth_branches = {}
      self._butruth_branches["TruthBToKMuMu_RecoIdx"] = self._gen_branches["GenPart_pdg"]



      # Add trigger decision to Bs candidates
      self._bu_branches["BToKMuMu_{}".format(self._trigger)] = np.repeat(self._event_branches[self._trigger], self._event_branches["nBToKMuMu"])

      # Print out length of arrays
      #for branch, array in self._bu_branches.items():
      #  print("{}\t{}".format(len(array.flatten()), branch))

      # flatten the jagged arrays to a normal numpy array, turn the whole dictionary to pandas dataframe
      self._bu_branches = pd.DataFrame.from_dict({branch: array.flatten() for branch, array in self._bu_branches.items()})

      # Minimum lepton pT
      self._bu_branches["BToKMuMu_fit_l_minpt"] = np.minimum(self._bu_branches["BToKMuMu_fit_l1_pt"], self._bu_branches["BToKMuMu_fit_l2_pt"])

      # general selection
      trigger_selection = self._bu_branches['BToKMuMu_{}'.format(self._trigger)]
      tag_selection = self._bu_branches["BToKMuMu_isTag"] & trigger_selection
      probe_selection = self._bu_branches["BToKMuMu_isProbe"] & trigger_selection

      sv_selection = (self._bu_branches['BToKMuMu_fit_pt'] > 3.0) \
                      & (np.abs(self._bu_branches['BToKMuMu_l_xy'] / self._bu_branches['BToKMuMu_l_xy_unc']) > 3.0 ) \
                      & (self._bu_branches['BToKMuMu_svprob'] > 0.1) \
                      & (self._bu_branches['BToKMuMu_fit_cos2D'] > 0.9)

      l1_selection = (self._bu_branches['BToKMuMu_fit_l1_pt'] > 1.5) \
                      & (np.abs(self._bu_branches['BToKMuMu_fit_l1_eta']) < 2.4)
      l2_selection = (self._bu_branches['BToKMuMu_fit_l2_pt'] > 1.5) \
                      & (np.abs(self._bu_branches['BToKMuMu_fit_l2_eta']) < 2.4)
      k_selection = (self._bu_branches['BToKMuMu_fit_k_pt'] > 0.5) \
                      & (np.abs(self._bu_branches['BToKMuMu_fit_k_eta']) < 2.5)

      jpsi_selection = (JPSI_1S_MASS - 0.2 < self._bu_branches['BToKMuMu_mll_fullfit']) & (self._bu_branches['BToKMuMu_mll_fullfit'] < JPSI_1S_MASS + 0.2)


      bu_selection = sv_selection & l1_selection & l2_selection & k_selection & jpsi_selection

      #print("N trigger_selection = {}".format(trigger_selection.sum()))
      #print("N tag_selection = {}".format(tag_selection.sum()))
      #print("N probe_selection = {}".format(probe_selection.sum()))
      #print("N sv_selection = {}".format(sv_selection.sum()))
      #print("N l1_selection = {}".format(l1_selection.sum()))
      #print("N l2_selection = {}".format(l2_selection.sum()))
      #print("N k_selection = {}".format(k_selection.sum()))
      #print("N jpsi_selection = {}".format(jpsi_selection.sum()))
      #print("N bu_selection = {}".format(bu_selection.sum()))

      '''

      if self._isMC:
        pass
        mc_matched_selection = (self._branches['BsToKKMuMu_l1_genPartIdx'] >= 0) \
                                & (self._branches['BsToKKMuMu_l2_genPartIdx'] >= 0) \
                                & (self._branches['BsToKKMuMu_k_genPartIdx'] >= 0)
        # B->K J/psi(ee)
        #mc_parent_selection = (abs(self._branches['BsToKKMuMu_l1_genMotherPdgId']) == 443) & (abs(self._branches['BsToKKMuMu_k_genMotherPdgId']) == 521)
        #mc_chain_selection = (self._branches['BsToKKMuMu_l1_genMotherPdgId'] == self._branches['BsToKKMuMu_l2_genMotherPdgId']) & (self._branches['BsToKKMuMu_k_genMotherPdgId'] == self._branches['BsToKKMuMu_l1Mother_genMotherPdgId']) & (self._branches['BsToKKMuMu_k_genMotherPdgId'] == self._branches['BsToKKMuMu_l2Mother_genMotherPdgId'])

        # B->K*(K pi) J/psi(ee)
        mc_parent_selection = (abs(self._branches['BsToKKMuMu_l1_genMotherPdgId']) == 443) & (abs(self._branches['BsToKKMuMu_k_genMotherPdgId']) == 313)
        mc_chain_selection = (self._branches['BsToKKMuMu_l1_genMotherPdgId'] == self._branches['BsToKKMuMu_l2_genMotherPdgId'])
        mc_selection = mc_matched_selection & mc_parent_selection & mc_chain_selection

      #additional_selection = b_sb_selection
      if self._isMC:
        selection = l1_selection & l2_selection & k_selection & mc_selection

      else:
        selection = l1_selection & l2_selection & k_selection
      '''

      for tag_type in ["inclusive", "triggered", "tag", "probe"]:
        this_selection = copy.deepcopy(bu_selection)
        if tag_type == "triggered":
          this_selection &= trigger_selection
        elif tag_type == "tag":
          this_selection &= tag_selection
        elif tag_type == "probe":
          this_selection &= probe_selection

        #print("tag_type {}".format(tag_type))
        #print("\tthis_selection.count = {}".format(this_selection.sum()))

        selected_branches = self._bu_branches[this_selection]

        fill_hist(self._histograms[tag_type]['BToKMuMu_chi2'], selected_branches['BToKMuMu_chi2'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_eta'], selected_branches['BToKMuMu_eta'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_fit_cos2D'], selected_branches['BToKMuMu_fit_cos2D'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_fit_eta'], selected_branches['BToKMuMu_fit_eta'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_fit_mass'], selected_branches['BToKMuMu_fit_mass'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_fit_phi'], selected_branches['BToKMuMu_fit_phi'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_fit_pt'], selected_branches['BToKMuMu_fit_pt'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_l_xy'], selected_branches['BToKMuMu_l_xy'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_l_xy_sig'], selected_branches['BToKMuMu_l_xy'].values / selected_branches['BToKMuMu_l_xy_unc'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_fit_l1_eta'], selected_branches['BToKMuMu_fit_l1_eta'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_fit_l1_phi'], selected_branches['BToKMuMu_fit_l1_phi'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_fit_l1_pt'], selected_branches['BToKMuMu_fit_l1_pt'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_fit_l2_eta'], selected_branches['BToKMuMu_fit_l2_eta'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_fit_l2_phi'], selected_branches['BToKMuMu_fit_l2_phi'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_fit_l2_pt'], selected_branches['BToKMuMu_fit_l2_pt'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_mass'], selected_branches['BToKMuMu_mass'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_mll_fullfit'], selected_branches['BToKMuMu_mll_fullfit'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_mll_llfit'], selected_branches['BToKMuMu_mll_llfit'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_mll_raw'], selected_branches['BToKMuMu_mll_raw'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_phi'], selected_branches['BToKMuMu_phi'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_pt'], selected_branches['BToKMuMu_pt'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_svprob'], selected_branches['BToKMuMu_svprob'].values)
        fill_hist(self._histograms[tag_type]['BToKMuMu_charge'], selected_branches['BToKMuMu_charge'].values)

        fill_hist(self._histograms[tag_type]['BToKMuMu_fit_l_minpt'], selected_branches["BToKMuMu_fit_l_minpt"].values)
      # End loop tag_Type
      
      # Debug absence of low-pT probes
      if ifile == 0:
        select_lowpt = (self._bu_branches["BToKMuMu_fit_pt"]  < 10.)
        select_lowpt_probe = (self._bu_branches["BToKMuMu_fit_pt"]  < 10.) & self._bu_branches["BToKMuMu_isProbe"]
        print("pT of Bus with pT<10 and isProbe")
        print(self._bu_branches["BToKMuMu_fit_pt"][select_lowpt_probe])
      
      # Cutflow
      cutflow_selection = np.ones_like(self._bu_branches["BToKMuMu_chi2"], dtype=int)
      self._cutflow_counts["Inclusive"] += cutflow_selection.sum()

      self._cutflow_counts[self._trigger] += trigger_selection.sum()

      # Inclusive branch
      cutflow_selection_inclusive = copy.deepcopy(cutflow_selection)
      cutflow_selection_inclusive &= sv_selection
      self._cutflow_counts["Inclusive SV"] += cutflow_selection_inclusive.sum()

      cutflow_selection_inclusive &= l1_selection & l2_selection & k_selection
      self._cutflow_counts["Inclusive mu-K"] += cutflow_selection_inclusive.sum()

      cutflow_selection_inclusive &= jpsi_selection
      self._cutflow_counts["Inclusive Jpsi"] += cutflow_selection_inclusive.sum()

      # Tag branch
      cutflow_selection_tag =  cutflow_selection & tag_selection
      self._cutflow_counts["Tag"] += cutflow_selection_tag.sum()

      cutflow_selection_tag = cutflow_selection_tag & sv_selection
      self._cutflow_counts["Tag SV"] += cutflow_selection_tag.sum()

      cutflow_selection_tag = cutflow_selection_tag & l1_selection & l2_selection & k_selection
      self._cutflow_counts["Tag mu-K"] += cutflow_selection_tag.sum()

      cutflow_selection_tag = cutflow_selection_tag & jpsi_selection
      self._cutflow_counts["Tag Jpsi"] += cutflow_selection_tag.sum()

      # Probe branch
      cutflow_selection_probe =  cutflow_selection & probe_selection
      self._cutflow_counts["Probe"] += cutflow_selection_probe.sum()

      cutflow_selection_probe = cutflow_selection_probe & sv_selection
      self._cutflow_counts["Probe SV"] += cutflow_selection_probe.sum()

      cutflow_selection_probe = cutflow_selection_probe & l1_selection & l2_selection & k_selection
      self._cutflow_counts["Probe mu-K"] += cutflow_selection_probe.sum()

      cutflow_selection_probe = cutflow_selection_probe & jpsi_selection
      self._cutflow_counts["Probe Jpsi"] += cutflow_selection_probe.sum()

    # End loop file
  # End run

  def finish(self):
    # Print cutflow
    for cut_name in self._cutflow_names:
      print "\t{}\t=>\t{}".format(cut_name, self._cutflow_counts[cut_name])

    self._output_file.cd()
    for tag_type in ["inclusive", "triggered", "tag", "probe"]:
      for hname, hist in self._histograms[tag_type].items():
        hist.write()

    for hname, hist in self._mu_histograms.items():
      hist.write()
