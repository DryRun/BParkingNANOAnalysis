#! /usr/bin/env python

#import ROOT
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

from BParkingNANOAnalysis.BParkingNANOAnalyzer.BaseAnalyzer import BParkingNANOAnalyzer
from BParkingNANOAnalysis.BParkingNANOAnalyzer.constants import *


class DataAnalyzer(BParkingNANOAnalyzer):
  def __init__(self, input_files, output_file):
    super(DataAnalyzer, self).__init__(input_files, output_file, False)
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
    self._bs_branchnames = [
      "BToPhiMuMu_chi2",
      "BToPhiMuMu_cos2D",
      "BToPhiMuMu_eta",
      "BToPhiMuMu_etaphi_fullfit",
      "BToPhiMuMu_fit_cos2D",
      "BToPhiMuMu_fit_eta",
      "BToPhiMuMu_fit_mass",
      "BToPhiMuMu_fit_massErr",
      "BToPhiMuMu_fit_phi",
      "BToPhiMuMu_fit_pt",
      "BToPhiMuMu_l_xy",
      "BToPhiMuMu_l_xy_unc",
      "BToPhiMuMu_lep1eta_fullfit",
      "BToPhiMuMu_lep1phi_fullfit",
      "BToPhiMuMu_lep1pt_fullfit",
      "BToPhiMuMu_lep2eta_fullfit",
      "BToPhiMuMu_lep2phi_fullfit",
      "BToPhiMuMu_lep2pt_fullfit",
      "BToPhiMuMu_mass",
      "BToPhiMuMu_max_dr",
      "BToPhiMuMu_min_dr",
      "BToPhiMuMu_mll_fullfit",
      "BToPhiMuMu_mll_llfit",
      "BToPhiMuMu_mll_raw",
      "BToPhiMuMu_mphi_fullfit",
      "BToPhiMuMu_phi",
      "BToPhiMuMu_phiphi_fullfit",
      "BToPhiMuMu_pt",
      "BToPhiMuMu_ptphi_fullfit",
      "BToPhiMuMu_svprob",
      "BToPhiMuMu_trk1eta_fullfit",
      "BToPhiMuMu_trk1phi_fullfit",
      "BToPhiMuMu_trk1pt_fullfit",
      "BToPhiMuMu_trk2eta_fullfit",
      "BToPhiMuMu_trk2phi_fullfit",
      "BToPhiMuMu_trk2pt_fullfit",
      "BToPhiMuMu_charge",
      "BToPhiMuMu_l1_idx",
      "BToPhiMuMu_l2_idx",
      "BToPhiMuMu_pdgId",
      "BToPhiMuMu_phi_idx",
      "BToPhiMuMu_trk1_idx",
      "BToPhiMuMu_trk2_idx",
    ]    
    self._event_branchnames = [
      "nBToKMuMu",
      "nBToPhiMuMu",
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
      #"nTrigObj",
      #"TrigObj_pt",
      #"TrigObj_eta",
      #"TrigObj_phi",
      #"TrigObj_l1pt",
      #"TrigObj_l1pt_2",
      #"TrigObj_l2pt",
      #"TrigObj_id",
      #"TrigObj_l1iso",
      #"TrigObj_l1charge",
      #"TrigObj_filterBits",
      #"nTriggerMuon",
      #"TriggerMuon_eta",
      #"TriggerMuon_mass",
      #"TriggerMuon_phi",
      #"TriggerMuon_pt",
      #"TriggerMuon_vx",
      #"TriggerMuon_vy",
      #"TriggerMuon_vz",
      #"TriggerMuon_charge",
      #"TriggerMuon_pdgId",
      #"TriggerMuon_trgMuonIndex",
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

    #if self._isMC:
    #  self._input_branches.extend(['GenPart_pdgId', 'GenPart_genPartIdxMother'])

    self._histograms = {}
    for tag_type in ["tag", "probe"]:
      self._histograms[tag_type] = {}
      for trig_type in ["trigpass", "triginclusive"]:
        self._histograms[tag_type][trig_type] = {}

        self._histograms[tag_type][trig_type]["BToKMuMu_chi2"]            = Hist(100, 0.0, 100.0, name="{}_{}_BToKMuMu_chi2".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_eta"]             = Hist(50, -5.0, 5.0, name="{}_{}_BToKMuMu_eta".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_fit_cos2D"]       = Hist(100, -1., 1., name="{}_{}_BToKMuMu_fit_cos2D".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_fit_eta"]         = Hist(50, -5.0, 5.0, name="{}_{}_BToKMuMu_fit_eta".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_fit_mass"]        = Hist(100, BU_MASS * 0.9, BU_MASS * 1.1, name="{}_{}_BToKMuMu_fit_mass".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_fit_phi"]         = Hist(50, -2.0*math.pi, 2.0*math.pi, name="{}_{}_BToKMuMu_fit_phi".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_fit_pt"]          = Hist(100, 0.0, 100.0, name="{}_{}_BToKMuMu_fit_pt".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_l_xy"]            = Hist(50, -1.0, 4.0, name="{}_{}_BToKMuMu_l_xy".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_l_xy_sig"]        = Hist(50, -1.0, 4.0, name="{}_{}_BToKMuMu_l_xy_sig".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_fit_l1_eta"]      = Hist(50, -5.0, 5.0, name="{}_{}_BToKMuMu_fit_l1_eta".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_fit_l1_phi"]      = Hist(50, -2.0*math.pi, 2.0*math.pi, name="{}_{}_BToKMuMu_fit_l1_phi".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_fit_l1_pt"]       = Hist(100, 0.0, 100.0, name="{}_{}_BToKMuMu_fit_l1_pt".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_fit_l2_eta"]      = Hist(50, -5.0, 5.0, name="{}_{}_BToKMuMu_fit_l2_eta".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_fit_l2_phi"]      = Hist(50, -2.0*math.pi, 2.0*math.pi, name="{}_{}_BToKMuMu_fit_l2_phi".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_fit_l2_pt"]       = Hist(100, 0.0, 100.0, name="{}_{}_BToKMuMu_fit_l2_pt".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_mass"]            = Hist(500, BU_MASS * 0.9, BU_MASS * 1.1, name="{}_{}_BToKMuMu_mass".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_mll_fullfit"]     = Hist(500, JPSI_1S_MASS * 0.9, JPSI_1S_MASS * 1.1, name="{}_{}_BToKMuMu_mll_fullfit".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_mll_llfit"]       = Hist(500, JPSI_1S_MASS * 0.9, JPSI_1S_MASS * 1.1, name="{}_{}_BToKMuMu_mll_llfit".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_mll_raw"]         = Hist(500, JPSI_1S_MASS * 0.9, JPSI_1S_MASS * 1.1, name="{}_{}_BToKMuMu_mll_raw".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_phi"]             = Hist(50, -2.0*math.pi, 2.0*math.pi, name="{}_{}_BToKMuMu_phi".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_pt"]              = Hist(100, 0.0, 100.0, name="{}_{}_BToKMuMu_pt".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_svprob"]          = Hist(100, -1.0, 1.0, name="{}_{}_BToKMuMu_svprob".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type]["BToKMuMu_charge"]          = Hist(3, -1.5, 1.5, name="{}_{}_BToKMuMu_charge".format(tag_type, trig_type), title="", type="F")
        self._histograms[tag_type][trig_type["BToKMuMu_fit_mass_pt"]]     = Hist2D(500, BU_MASS * 0.9, BU_MASS * 1.1, 200, 0., 100.)

        self._histograms[tag_type][trig]["BToPhiMuMu_chi2"]                 = Hist(100, 0.0, 100.0, name="{}_{}_BToPhiMuMu_chi2".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_eta"]                  = Hist(50, -5.0, 5.0, name="{}_{}_BToPhiMuMu_eta".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_fit_cos2D"]            = Hist(100, -1., 1., name="{}_{}_BToPhiMuMu_fit_cos2D".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_fit_eta"]              = Hist(50, -5.0, 5.0, name="{}_{}_BToPhiMuMu_fit_eta".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_fit_mass"]             = Hist(100, BS_MASS * 0.9, BS_MASS * 1.1, name="{}_{}_BToPhiMuMu_fit_mass".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_fit_phi"]              = Hist(50, -2.0*math.pi, 2.0*math.pi, name="{}_{}_BToPhiMuMu_fit_phi".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_fit_pt"]               = Hist(100, 0.0, 100.0, name="{}_{}_BToPhiMuMu_fit_pt".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_l_xy"]                 = Hist(50, -1.0, 4.0, name="{}_{}_BToPhiMuMu_l_xy".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_l_xy_sig"]             = Hist(50, -1.0, 4.0, name="{}_{}_BToPhiMuMu_l_xy_sig".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_lep1eta_fullfit"]      = Hist(50, -5.0, 5.0, name="{}_{}_BToPhiMuMu_lep1eta_fullfit".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_lep1phi_fullfit"]      = Hist(50, -2.0*math.pi, 2.0*math.pi, name="{}_{}_BToPhiMuMu_lep1phi_fullfit".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_lep1pt_fullfit"]       = Hist(100, 0.0, 100.0, name="{}_{}_BToPhiMuMu_lep1pt_fullfit".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_lep2eta_fullfit"]      = Hist(50, -5.0, 5.0, name="{}_{}_BToPhiMuMu_lep2eta_fullfit".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_lep2phi_fullfit"]      = Hist(50, -2.0*math.pi, 2.0*math.pi, name="{}_{}_BToPhiMuMu_lep2phi_fullfit".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_lep2pt_fullfit"]       = Hist(100, 0.0, 100.0, name="{}_{}_BToPhiMuMu_lep2pt_fullfit".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_mass"]                 = Hist(500, BS_MASS * 0.9, BS_MASS * 1.1, name="{}_{}_BToPhiMuMu_mass".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_mll_fullfit"]          = Hist(500, JPSI_1S_MASS * 0.9, JPSI_1S_MASS * 1.1, name="{}_{}_BToPhiMuMu_mll_fullfit".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_mll_llfit"]            = Hist(500, JPSI_1S_MASS * 0.9, JPSI_1S_MASS * 1.1, name="{}_{}_BToPhiMuMu_mll_llfit".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_mll_raw"]              = Hist(500, JPSI_1S_MASS * 0.9, JPSI_1S_MASS * 1.1, name="{}_{}_BToPhiMuMu_mll_raw".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_mphi_fullfit"]         = Hist(500, PHI_1020_MASS * 0.9, PHI_1020_MASS * 1.1, name="{}_{}_BToPhiMuMu_mphi_fullfit".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_phi"]                  = Hist(50, -2.0*math.pi, 2.0*math.pi, name="{}_{}_BToPhiMuMu_phi".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_etaphi_fullfit"]       = Hist(50, -5.0, 5.0, name="{}_{}_BToPhiMuMu_etaphi_fullfit".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_phiphi_fullfit"]       = Hist(50, -2.0*math.pi, 2.0*math.pi, name="{}_{}_BToPhiMuMu_phiphi_fullfit".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_pt"]                   = Hist(100, 0.0, 100.0, name="{}_{}_BToPhiMuMu_pt".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_ptphi_fullfit"]        = Hist(100, 0.0, 100.0, name="{}_{}_BToPhiMuMu_ptphi_fullfit".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_svprob"]               = Hist(100, -1.0, 1.0, name="{}_{}_BToPhiMuMu_svprob".format(tag_type, trig), title="", type="F")
        self._histograms[tag_type][trig]["BToPhiMuMu_charge"]               = Hist(3, -1.5, 1.5, name="{}_{}_BToPhiMuMu_charge".format(tag_type, trig), title="", type="F")


  def run(self):
    print('[DataAnalyzer::run] INFO: Running the analyzer...')
    self.print_timestamp()
    for ifile, filename in enumerate(self._input_files):
      print('[DataAnalyzer::run] INFO: FILE: {}/{}. Getting branches from file...'.format(ifile, len(self._input_files)))
      tree = uproot.open(filename)['Events']
      self._bu_branches = {key: awkward.fromiter(branch) for key, branch in tree.arrays(self._bu_branchnames).items()}
      self._event_branches = {key: awkward.fromiter(branch) for key, branch in tree.arrays(self._event_branchnames).items()}
      self._muon_branches = {key: awkward.fromiter(branch) for key, branch in tree.arrays(self._muon_branchnames).items()}

      print('[DataAnalyzer::run] INFO: FILE: {}/{}. Analyzing...'.format(ifile, len(self._input_files)))

      # Tag/probe determination
      isTrig_mu1 = self._muon_branches["Muon_isTriggering"][self._bu_branches["BToKMuMu_l1Idx"]]
      isTrig_mu2 = self._muon_branches["Muon_isTriggering"][self._bu_branches["BToKMuMu_l2Idx"]]
      bu_trig_count = isTrig_mu1 + isTrig_mu2
      total_trig_count = self._muon_branches["Muon_isTriggering"].sum()
      tag_count = total_trig_count - bu_trig_count
      self._bu_branches["BToKMuMu_isTag"] = isTrig_mu1 | isTrig_mu2
      self._bu_branches["BToKMuMu_isProbe"] = (tag_count > 0)

      '''
      if self._isMC:
        # reconstruct full decay chain
        self._branches['BsToKKMuMu_l1_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['Electron_genPartIdx'][self._branches['BsToKKMuMu_l1Idx']]]
        self._branches['BsToKKMuMu_l2_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['Electron_genPartIdx'][self._branches['BsToKKMuMu_l2Idx']]]
        self._branches['BsToKKMuMu_k_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['ProbeTracks_genPartIdx'][self._branches['BsToKKMuMu_kIdx']]]

        self._branches['BsToKKMuMu_l1_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BsToKKMuMu_l1_genMotherIdx']]
        self._branches['BsToKKMuMu_l2_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BsToKKMuMu_l2_genMotherIdx']]
        self._branches['BsToKKMuMu_k_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BsToKKMuMu_k_genMotherIdx']]

        self._branches['BsToKKMuMu_l1Mother_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['BsToKKMuMu_l1_genMotherIdx']]
        self._branches['BsToKKMuMu_l2Mother_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['BsToKKMuMu_l2_genMotherIdx']]

        self._branches['BsToKKMuMu_l1Mother_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BsToKKMuMu_l1Mother_genMotherIdx']]
        self._branches['BsToKKMuMu_l2Mother_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BsToKKMuMu_l2Mother_genMotherIdx']]
      '''

      # Add trigger decision to Bs candidates
      self._bu_branches["BToKMuMu_{}".format(self._trigger)] = np.repeat(self._event_branches[self._trigger], self._event_branches["nBToKMuMu"])

      # Print out length of arrays
      for branch, array in self._bu_branches.items():
        print("{}\t{}".format(len(array.flatten()), branch))

      # flatten the jagged arrays to a normal numpy array, turn the whole dictionary to pandas dataframe
      self._bu_branches = pd.DataFrame.from_dict({branch: array.flatten() for branch, array in self._bu_branches.items()})

      # general selection
      tag_selection = self._bu_branches["BToKMuMu_isTag"]
      probe_selection = self._bu_branches["BToKMuMu_isProbe"]

      sv_selection = (self._bu_branches['BToKMuMu_fit_pt'] > 3.0) \
                      & (self._bu_branches['BToKMuMu_l_xy'] / self._bu_branches['BToKMuMu_l_xy_unc'] > 3.0 ) \
                      & (self._bu_branches['BToKMuMu_svprob'] > 0.1) \
                      #& (self._bu_branches['BsToKKMuMu_fit_cos2D'] > 0.9)

      l1_selection = (self._bu_branches['BToKMuMu_fit_l1_pt'] > 1.5) \
                      & (np.abs(self._bu_branches['BToKMuMu_fit_l1_eta']) < 2.4)
      l2_selection = (self._bu_branches['BToKMuMu_fit_l2_pt'] > 1.5) \
                      & (np.abs(self._bu_branches['BToKMuMu_fit_l2_eta']) < 2.4)
      k_selection = (self._bu_branches['BToKMuMu_fit_k_pt'] > 0.5) \
                      & (np.abs(self._bu_branches['BToKMuMu_fit_k_eta']) < 2.5)

      jpsi_selection = (JPSI_1S_MASS - 0.2 < self._bu_branches['BToKMuMu_mll_fullfit']) & (self._bu_branches['BToKMuMu_mll_fullfit'] < JPSI_1S_MASS + 0.2)


      bu_selection = sv_selection & l1_selection & l2_selection & k_selection & jpsi_selection \

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

      for tag_type in ["tag", "probe"]:
        if tag_type == "tag":
          this_selection = bu_selection & self._bu_branches["BToKMuMu_isTag"]
        else:
          this_selection = bu_selection & self._bu_branches["BToKMuMu_isProbe"]

        for trig_type in ["trigpass", "triginclusive"]:
          if trig_type == "trigpass":
            this_selection = this_selection & self._bu_branches['BToKMuMu_{}'.format(self._trigger)]

          selected_branches = self._bu_branches[this_selection]

          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_chi2'], selected_branches['BToKMuMu_chi2'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_eta'], selected_branches['BToKMuMu_eta'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_fit_cos2D'], selected_branches['BToKMuMu_fit_cos2D'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_fit_eta'], selected_branches['BToKMuMu_fit_eta'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_fit_mass'], selected_branches['BToKMuMu_fit_mass'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_fit_phi'], selected_branches['BToKMuMu_fit_phi'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_fit_pt'], selected_branches['BToKMuMu_fit_pt'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_l_xy'], selected_branches['BToKMuMu_l_xy'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_l_xy_sig'], selected_branches['BToKMuMu_l_xy'].values / selected_branches['BToKMuMu_l_xy_unc'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_fit_l1_eta'], selected_branches['BToKMuMu_fit_l1_eta'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_fit_l1_phi'], selected_branches['BToKMuMu_fit_l1_phi'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_fit_l1_pt'], selected_branches['BToKMuMu_fit_l1_pt'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_fit_l2_eta'], selected_branches['BToKMuMu_fit_l2_eta'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_fit_l2_phi'], selected_branches['BToKMuMu_fit_l2_phi'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_fit_l2_pt'], selected_branches['BToKMuMu_fit_l2_pt'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_mass'], selected_branches['BToKMuMu_mass'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_mll_fullfit'], selected_branches['BToKMuMu_mll_fullfit'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_mll_llfit'], selected_branches['BToKMuMu_mll_llfit'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_mll_raw'], selected_branches['BToKMuMu_mll_raw'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_phi'], selected_branches['BToKMuMu_phi'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_pt'], selected_branches['BToKMuMu_pt'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_svprob'], selected_branches['BToKMuMu_svprob'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToKMuMu_charge'], selected_branches['BToKMuMu_charge'].values)

  def finish(self):
    self._output_file.cd()
    for tag_type in ["tag", "probe"]:
      for trig_type in ["trigpass", "triginclusive"]:
        for hname, hist in self._histograms[tag_type][trig_type].items():
          hist.write()
