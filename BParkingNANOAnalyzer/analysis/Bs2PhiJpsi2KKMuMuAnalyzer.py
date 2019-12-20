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

from BParkingNANOAnalysis.BParkingNANOAnalyzer.BaseAnalyzer import BParkingNANOAnalyzer
from BParkingNANOAnalysis.BParkingNANOAnalyzer.constants import *

np.set_printoptions(threshold=sys.maxsize)


class Bs2PhiJpsi2KKMuMuAnalyzer(BParkingNANOAnalyzer):
  def __init__(self, input_files, output_file, isMC=False):
    super(Bs2PhiJpsi2KKMuMuAnalyzer, self).__init__(input_files, output_file, isMC)
    self._trigger = "HLT_Mu7_IP4"

  def start(self):
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
      #"nPhi",
      #"Phi_eta",
      #"Phi_fitted_eta",
      #"Phi_fitted_mass",
      #"Phi_fitted_phi",
      #"Phi_fitted_pt",
      #"Phi_mass",
      #"Phi_phi",
      #"Phi_pt",
      #"Phi_svprob",
      #"Phi_trk_deltaR",
      #"Phi_charge",
      #"Phi_pdgId",
      #"Phi_trk1_idx",
      #"Phi_trk2_idx",
      "nMuon",
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
      #"L1_SingleMu7er1p5",
      #"L1_SingleMu8er1p5",
      #"L1_SingleMu9er1p5",
      #"L1_SingleMu10er1p5",
      #"L1_SingleMu12er1p5",
      #"L1_SingleMu22",
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
    self._mu_histograms = {}
    self._mu_histograms["nMuon"]          = Hist(11,-0.5, 10.5, name="nMuon", title="", type="F")
    self._mu_histograms["nMuon_isTrig"]   = Hist(11,-0.5, 10.5, name="nMuon_isTrig", title="", type="F")
    self._mu_histograms["Muon_pt"]        = Hist(100, 0.0, 100.0, name="Muon_pt", title="", type="F")
    self._mu_histograms["Muon_pt_isTrig"] = Hist(100, 0.0, 100.0, name="Muon_pt_isTrig", title="", type="F")

    self._histograms = {}
    for tag_type in ["tag", "probe"]:
      self._histograms[tag_type] = {}
      for trig in ["trigpass", "triginclusive"]:
        self._histograms[tag_type][trig] = {}
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

    self._cutflow_names = []
    self._cutflow_names.append("Inclusive")
    self._cutflow_names.append(self._trigger)

    self._cutflow_names.append("Tag")
    self._cutflow_names.append("Tag SV")
    self._cutflow_names.append("Tag mu-K")
    self._cutflow_names.append("Tag Jpsi")
    self._cutflow_names.append("Tag phi")

    self._cutflow_names.append("Probe")
    self._cutflow_names.append("Probe SV")
    self._cutflow_names.append("Probe mu-K")
    self._cutflow_names.append("Probe Jpsi")
    self._cutflow_names.append("Probe phi")

    self._cutflow_counts = {}
    for name in self._cutflow_names:
      self._cutflow_counts[name] = 0

  def run(self):
    print('[Bs2PhiJpsi2KKMuMuAnalyzer::run] INFO: Running the analyzer...')
    self.print_timestamp()
    for ifile, filename in enumerate(self._input_files):
      print('[Bs2PhiJpsi2KKMuMuAnalyzer::run] INFO: FILE: {}/{}. Getting branches from file...'.format(ifile, len(self._input_files)))
      tree = uproot.open(filename)['Events']
      self._bs_branches = {key: awkward.fromiter(branch) for key, branch in tree.arrays(self._bs_branchnames).items()}
      self._event_branches = {key: awkward.fromiter(branch) for key, branch in tree.arrays(self._event_branchnames).items()}
      self._muon_branches = {key: awkward.fromiter(branch) for key, branch in tree.arrays(self._muon_branchnames).items()}

      print('[Bs2PhiJpsi2KKMuMuAnalyzer::run] INFO: FILE: {}/{}. Analyzing...'.format(ifile, len(self._input_files)))

      # Muon information
      fill_hist(self._mu_histograms["nMuon"], self._event_branches["nMuon"])
      fill_hist(self._mu_histograms["nMuon_isTrig"], self._muon_branches["Muon_pt"][self._muon_branches["Muon_isTriggering"]].count())
      fill_hist(self._mu_histograms["Muon_pt"], self._muon_branches["Muon_pt"].flatten())
      fill_hist(self._mu_histograms["Muon_pt_isTrig"], self._muon_branches["Muon_pt"][self._muon_branches["Muon_isTriggering"]].flatten())


      # Tag/probe determination
      isTrig_mu1 = self._muon_branches["Muon_isTriggering"][self._bs_branches["BToPhiMuMu_l1_idx"]]
      isTrig_mu2 = self._muon_branches["Muon_isTriggering"][self._bs_branches["BToPhiMuMu_l2_idx"]]
      bs_trig_count = isTrig_mu1 + isTrig_mu2
      total_trig_count = self._muon_branches["Muon_isTriggering"].sum()
      tag_count = total_trig_count - bs_trig_count
      self._bs_branches["BToPhiMuMu_isTag"] = isTrig_mu1 | isTrig_mu2
      self._bs_branches["BToPhiMuMu_isProbe"] = (tag_count >= 1)

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

      # Associate muons to trigger muons
      '''
      Relevant branches:
      "BToPhiMuMu_l1_idx",
      "BToPhiMuMu_l2_idx",
      '''

      # Add trigger decision to Bs candidates
      self._bs_branches["BToPhiMuMu_{}".format(self._trigger)] = np.repeat(self._event_branches[self._trigger], self._event_branches["nBToPhiMuMu"])

      # Print out length of arrays
      #for branch, array in self._bs_branches.items():
      #  print("{}\t{}".format(len(array.flatten()), branch))

      # flatten the jagged arrays to a normal numpy array, turn the whole dictionary to pandas dataframe
      self._bs_branches = pd.DataFrame.from_dict({branch: array.flatten() for branch, array in self._bs_branches.items()})

      # general selection
      tag_selection = self._bs_branches["BToPhiMuMu_isTag"]
      probe_selection = self._bs_branches["BToPhiMuMu_isProbe"]

      sv_selection = (self._bs_branches['BToPhiMuMu_fit_pt'] > 3.0) \
                      & (np.abs(self._bs_branches['BToPhiMuMu_l_xy'] / self._bs_branches['BToPhiMuMu_l_xy_unc']) > 3.0 ) \
                      & (self._bs_branches['BToPhiMuMu_svprob'] > 0.1) \
                      & (self._bs_branches['BToPhiMuMu_fit_cos2D'] > 0.9)

      l1_selection = (self._bs_branches['BToPhiMuMu_lep1pt_fullfit'] > 1.5) \
                      & (np.abs(self._bs_branches['BToPhiMuMu_lep1eta_fullfit']) < 2.4)
      l2_selection = (self._bs_branches['BToPhiMuMu_lep2pt_fullfit'] > 1.5) \
                      & (np.abs(self._bs_branches['BToPhiMuMu_lep2eta_fullfit']) < 2.4)
      k1_selection = (self._bs_branches['BToPhiMuMu_trk1pt_fullfit'] > 0.5) \
                      & (np.abs(self._bs_branches['BToPhiMuMu_trk1eta_fullfit']) < 2.5)
      k2_selection = (self._bs_branches['BToPhiMuMu_trk2pt_fullfit'] > 0.5) \
                      & (np.abs(self._bs_branches['BToPhiMuMu_trk1eta_fullfit']) < 2.5)

      jpsi_selection = (JPSI_1S_MASS - 0.2 < self._bs_branches['BToPhiMuMu_mll_fullfit']) & (self._bs_branches['BToPhiMuMu_mll_fullfit'] < JPSI_1S_MASS + 0.2)

      phi_selection = (PHI_1020_MASS - 0.2 < self._bs_branches['BToPhiMuMu_mphi_fullfit']) & (self._bs_branches['BToPhiMuMu_mphi_fullfit'] < PHI_1020_MASS + 0.2)

      bs_selection = sv_selection & l1_selection & l2_selection & k1_selection & k2_selection & jpsi_selection & phi_selection \

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
          this_selection = bs_selection & self._bs_branches["BToPhiMuMu_isTag"]
        else:
          this_selection = bs_selection & self._bs_branches["BToPhiMuMu_isProbe"]

        for trig_type in ["trigpass", "triginclusive"]:
          if trig_type == "trigpass":
            this_selection = this_selection & self._bs_branches['BToPhiMuMu_{}'.format(self._trigger)]

          selected_branches = self._bs_branches[this_selection]

          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_chi2'], selected_branches['BToPhiMuMu_chi2'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_eta'], selected_branches['BToPhiMuMu_eta'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_fit_cos2D'], selected_branches['BToPhiMuMu_fit_cos2D'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_fit_eta'], selected_branches['BToPhiMuMu_fit_eta'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_fit_mass'], selected_branches['BToPhiMuMu_fit_mass'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_fit_phi'], selected_branches['BToPhiMuMu_fit_phi'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_fit_pt'], selected_branches['BToPhiMuMu_fit_pt'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_l_xy'], selected_branches['BToPhiMuMu_l_xy'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_l_xy_sig'], selected_branches['BToPhiMuMu_l_xy'].values / selected_branches['BToPhiMuMu_l_xy_unc'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_lep1eta_fullfit'], selected_branches['BToPhiMuMu_lep1eta_fullfit'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_lep1phi_fullfit'], selected_branches['BToPhiMuMu_lep1phi_fullfit'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_lep1pt_fullfit'], selected_branches['BToPhiMuMu_lep1pt_fullfit'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_lep2eta_fullfit'], selected_branches['BToPhiMuMu_lep2eta_fullfit'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_lep2phi_fullfit'], selected_branches['BToPhiMuMu_lep2phi_fullfit'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_lep2pt_fullfit'], selected_branches['BToPhiMuMu_lep2pt_fullfit'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_mass'], selected_branches['BToPhiMuMu_mass'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_mll_fullfit'], selected_branches['BToPhiMuMu_mll_fullfit'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_mll_llfit'], selected_branches['BToPhiMuMu_mll_llfit'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_mll_raw'], selected_branches['BToPhiMuMu_mll_raw'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_mphi_fullfit'], selected_branches['BToPhiMuMu_mphi_fullfit'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_phi'], selected_branches['BToPhiMuMu_phi'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_etaphi_fullfit'], selected_branches['BToPhiMuMu_etaphi_fullfit'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_phiphi_fullfit'], selected_branches['BToPhiMuMu_phiphi_fullfit'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_pt'], selected_branches['BToPhiMuMu_pt'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_ptphi_fullfit'], selected_branches['BToPhiMuMu_ptphi_fullfit'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_svprob'], selected_branches['BToPhiMuMu_svprob'].values)
          fill_hist(self._histograms[tag_type][trig_type]['BToPhiMuMu_charge'], selected_branches['BToPhiMuMu_charge'].values)

      # Cutflow
      self._cutflow_selection = np.ones_like(self._bs_branches["BToPhiMuMu_chi2"], dtype=int)
      self._cutflow_counts["Inclusive"] += self._cutflow_selection.sum()

      self._cutflow_selection = self._cutflow_selection & self._bs_branches['BToPhiMuMu_{}'.format(self._trigger)] 
      self._cutflow_counts[self._trigger] += self._cutflow_selection.sum()

      # Fork cutflow into tag and probe
      self._cutflow_selection_tag =  self._cutflow_selection & tag_selection
      self._cutflow_counts["Tag"] += self._cutflow_selection_tag.sum()

      self._cutflow_selection_tag = self._cutflow_selection_tag & sv_selection
      self._cutflow_counts["Tag SV"] += self._cutflow_selection_tag.sum()

      self._cutflow_selection_tag = self._cutflow_selection_tag & l1_selection & l2_selection & k1_selection & k2_selection
      self._cutflow_counts["Tag mu-K"] += self._cutflow_selection_tag.sum()

      self._cutflow_selection_tag = self._cutflow_selection_tag & jpsi_selection
      self._cutflow_counts["Tag Jpsi"] += self._cutflow_selection_tag.sum()

      self._cutflow_selection_tag = self._cutflow_selection_tag & phi_selection
      self._cutflow_counts["Tag phi"] += self._cutflow_selection_tag.sum()

      # Probe
      self._cutflow_selection_probe =  self._cutflow_selection & probe_selection
      self._cutflow_counts["Probe"] += self._cutflow_selection_probe.sum()

      self._cutflow_selection_probe = self._cutflow_selection_probe & sv_selection
      self._cutflow_counts["Probe SV"] += self._cutflow_selection_probe.sum()

      self._cutflow_selection_probe = self._cutflow_selection_probe & l1_selection & l2_selection & k1_selection & k2_selection
      self._cutflow_counts["Probe mu-K"] += self._cutflow_selection_probe.sum()

      self._cutflow_selection_probe = self._cutflow_selection_probe & jpsi_selection
      self._cutflow_counts["Probe Jpsi"] += self._cutflow_selection_probe.sum()

      self._cutflow_selection_probe = self._cutflow_selection_probe & phi_selection
      self._cutflow_counts["Probe phi"] += self._cutflow_selection_probe.sum()

  def finish(self):
    # Print cutflow
    for cut_name in self._cutflow_names:
      print "\t{}\t=>\t{}".format(cut_name, self._cutflow_counts[cut_name])

    self._output_file.cd()
    for tag_type in ["tag", "probe"]:
      for trig_type in ["trigpass", "triginclusive"]:
        for hname, hist in self._histograms[tag_type][trig_type].items():
          hist.write()

    for hname, hist in self._mu_histograms.items():
      hist.write()
