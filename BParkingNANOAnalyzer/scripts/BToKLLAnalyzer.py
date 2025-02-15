#! /usr/bin/env python

#import ROOT
from math import ceil
import awkward
import uproot
import uproot_methods
import pandas as pd
import numpy as np
from BParkingNANOAnalysis.BParkingNANOAnalyzer.BaseAnalyzer import BParkingNANOAnalyzer


class BToKLLAnalyzer(BParkingNANOAnalyzer):
  def __init__(self, inputfiles, outputfile, hist=False, isMC=False):
    self._isMC = isMC
    inputbranches_BToKEE = ['nBToKEE',
                            'BToKEE_mll_raw',
                            'BToKEE_mll_llfit',
                            'BToKEE_mllErr_llfit',
                            'BToKEE_mll_fullfit',
                            'BToKEE_mass',
                            'BToKEE_fit_mass',
                            'BToKEE_fit_massErr',
                            'BToKEE_l1Idx',
                            'BToKEE_l2Idx',
                            'BToKEE_kIdx',
                            'BToKEE_l_xy',
                            'BToKEE_l_xy_unc',
                            'BToKEE_fit_pt',
                            'BToKEE_fit_eta',
                            'BToKEE_fit_phi',
                            'BToKEE_fit_l1_pt',
                            'BToKEE_fit_l1_eta',
                            'BToKEE_fit_l1_phi',
                            'BToKEE_fit_l2_pt',
                            'BToKEE_fit_l2_eta',
                            'BToKEE_fit_l2_phi',
                            'BToKEE_fit_k_pt',
                            'BToKEE_fit_k_eta',
                            'BToKEE_fit_k_phi',
                            'BToKEE_svprob',
                            'BToKEE_fit_cos2D',
                            'Electron_pt',
                            'Electron_eta',
                            'Electron_phi',
                            'Electron_dz',
                            'Electron_dxy',
                            'Electron_dxyErr',
                            'Electron_ptBiased',
                            'Electron_unBiased',
                            'Electron_convVeto',
                            'Electron_isLowPt',
                            'Electron_isPF',
                            'Electron_isPFoverlap',
                            'Electron_mvaId',
                            'Electron_lostHits',
                            'Electron_genPartIdx',
                            'ProbeTracks_pt',
                            'ProbeTracks_DCASig',
                            'ProbeTracks_eta',
                            'ProbeTracks_phi',
                            'ProbeTracks_dz',
                            #'ProbeTracks_isLostTrk',
                            #'ProbeTracks_isPacked',
                            'ProbeTracks_genPartIdx',
                            #'HLT_Mu9_IP6_*',
                            'event'
                            ]

    inputbranches_BToKEE_mc = ['GenPart_pdgId',
                               'GenPart_genPartIdxMother'
                               ]

    if self._isMC:
      inputbranches_BToKEE += inputbranches_BToKEE_mc

    
    outputbranches_BToKEE = {'BToKEE_mll_raw': {'nbins': 50, 'xmin': 0.0, 'xmax': 5.0},
                             'BToKEE_mll_llfit': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                             'BToKEE_mllErr_llfit': {'nbins': 30, 'xmin': 0.0, 'xmax': 3.0},
                             'BToKEE_mll_fullfit': {'nbins': 30, 'xmin': 2.6, 'xmax': 3.6},
                             'BToKEE_mass': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                             'BToKEE_fit_mass': {'nbins': 30, 'xmin': 4.7, 'xmax': 6.0},
                             'BToKEE_fit_massErr': {'nbins': 30, 'xmin': 0.0, 'xmax': 3.0},
                             'BToKEE_fit_l1_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                             'BToKEE_fit_l2_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                             'BToKEE_fit_l1_normpt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                             'BToKEE_fit_l2_normpt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                             'BToKEE_fit_l1_eta': {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0},
                             'BToKEE_fit_l2_eta': {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0},
                             'BToKEE_fit_l1_phi': {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0},
                             'BToKEE_fit_l2_phi': {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0},
                             'BToKEE_l1_dxy_sig': {'nbins': 50, 'xmin': -30.0, 'xmax': 30.0},
                             'BToKEE_l2_dxy_sig': {'nbins': 50, 'xmin': -30.0, 'xmax': 30.0},
                             'BToKEE_l1_dz': {'nbins': 50, 'xmin': -1.0, 'xmax': 1.0},
                             'BToKEE_l2_dz': {'nbins': 50, 'xmin': -1.0, 'xmax': 1.0},
                             'BToKEE_l1_unBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                             'BToKEE_l2_unBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                             'BToKEE_l1_ptBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                             'BToKEE_l2_ptBiased': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                             'BToKEE_l1_mvaId': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                             'BToKEE_l2_mvaId': {'nbins': 50, 'xmin': -2.0, 'xmax': 10.0},
                             'BToKEE_l1_isPF': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                             'BToKEE_l2_isPF': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                             'BToKEE_l1_isLowPt': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                             'BToKEE_l2_isLowPt': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                             'BToKEE_l1_isPFoverlap': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                             'BToKEE_l2_isPFoverlap': {'nbins': 2, 'xmin': 0, 'xmax': 2},
                             'BToKEE_fit_k_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                             'BToKEE_fit_k_normpt': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                             'BToKEE_fit_k_eta': {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0},
                             'BToKEE_fit_k_phi': {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0},
                             'BToKEE_k_DCASig': {'nbins': 50, 'xmin': 0.0, 'xmax': 10.0},
                             'BToKEE_fit_pt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                             'BToKEE_fit_eta': {'nbins': 50, 'xmin': -3.0, 'xmax': 3.0},
                             'BToKEE_fit_phi': {'nbins': 50, 'xmin': -4.0, 'xmax': 4.0},
                             'BToKEE_fit_normpt': {'nbins': 50, 'xmin': 0.0, 'xmax': 30.0},
                             'BToKEE_svprob': {'nbins': 50, 'xmin': 0.0, 'xmax': 1.0},
                             'BToKEE_fit_cos2D': {'nbins': 50, 'xmin': 0.999, 'xmax': 1.0},
                             'BToKEE_l_xy_sig': {'nbins': 50, 'xmin': 0.0, 'xmax': 50.0},
                             'BToKEE_event': {'nbins': 10, 'xmin': 0, 'xmax': 10},
                             }

    super(BToKLLAnalyzer, self).__init__(inputfiles, outputfile, inputbranches_BToKEE, outputbranches_BToKEE, hist)

  def run(self):
    ELECTRON_MASS = 0.000511
    K_MASS = 0.493677
    JPSI_LOW = 2.9
    JPSI_UP = 3.3
    B_MASS = 5.245
    B_SIGMA = 9.155e-02
    B_LOWSB_LOW = B_MASS - 6.0*B_SIGMA
    B_LOWSB_UP = B_MASS - 3.0*B_SIGMA
    B_UPSB_LOW = B_MASS + 3.0*B_SIGMA
    B_UPSB_UP = B_MASS + 6.0*B_SIGMA
    B_LOW = 4.5
    B_UP = 6.0

    print('[BToKLLAnalyzer::run] INFO: Running the analyzer...')
    self.print_timestamp()
    self.init_output()
    for (self._ifile, filename) in enumerate(self._file_in_name):
      print('[BToKLLAnalyzer::run] INFO: FILE: {}/{}. Loading file...'.format(self._ifile+1, self._num_files))
      tree = uproot.open(filename)['Events']
      self._branches = tree.arrays(self._inputbranches)
      self._branches = {key: awkward.fromiter(branch) for key, branch in self._branches.items()}

      print('[BToKLLAnalyzer::run] INFO: FILE: {}/{}. Analyzing...'.format(self._ifile+1, self._num_files))

      if self._isMC:
        # reconstruct full decay chain
        self._branches['BToKEE_l1_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['Electron_genPartIdx'][self._branches['BToKEE_l1Idx']]]
        self._branches['BToKEE_l2_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['Electron_genPartIdx'][self._branches['BToKEE_l2Idx']]]
        self._branches['BToKEE_k_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['ProbeTracks_genPartIdx'][self._branches['BToKEE_kIdx']]]

        self._branches['BToKEE_l1_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToKEE_l1_genMotherIdx']]
        self._branches['BToKEE_l2_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToKEE_l2_genMotherIdx']]
        self._branches['BToKEE_k_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToKEE_k_genMotherIdx']]

        self._branches['BToKEE_l1Mother_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['BToKEE_l1_genMotherIdx']]
        self._branches['BToKEE_l2Mother_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['BToKEE_l2_genMotherIdx']]

        self._branches['BToKEE_l1Mother_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToKEE_l1Mother_genMotherIdx']]
        self._branches['BToKEE_l2Mother_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BToKEE_l2Mother_genMotherIdx']]


      # remove cross referencing
      for branch in self._branches.keys():
        if 'Electron_' in branch:
          self._branches['BToKEE_l1_'+branch.replace('Electron_','')] = self._branches[branch][self._branches['BToKEE_l1Idx']] 
          self._branches['BToKEE_l2_'+branch.replace('Electron_','')] = self._branches[branch][self._branches['BToKEE_l2Idx']] 
          del self._branches[branch]

        if 'ProbeTracks_' in branch:
          self._branches['BToKEE_k_'+branch.replace('ProbeTracks_','')] = self._branches[branch][self._branches['BToKEE_kIdx']] 
          del self._branches[branch]
        
        if 'GenPart_' in branch:
          del self._branches[branch]

        if 'HLT_Mu9_IP6_' in branch:
          self._branches['BToKEE_'+branch] = np.repeat(self._branches[branch], self._branches['nBToKEE'])
          del self._branches[branch]

        if branch == 'event':
          self._branches['BToKEE_'+branch] = np.repeat(self._branches[branch], self._branches['nBToKEE'])
          del self._branches[branch]
        

      del self._branches['nBToKEE']



      #l1_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_l1_pt'], self._branches['BToKEE_l1_eta'], self._branches['BToKEE_l1_phi'], ELECTRON_MASS)
      #l2_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_l2_pt'], self._branches['BToKEE_l2_eta'], self._branches['BToKEE_l2_phi'], ELECTRON_MASS)
      #k_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BToKEE_k_pt'], self._branches['BToKEE_k_eta'], self._branches['BToKEE_k_phi'], K_MASS)
      #self._branches['BToKEE_mll_raw'] = (l1_p4 + l2_p4).mass
      #self._branches['BToKEE_mass'] = (l1_p4 + l2_p4 + k_p4).mass

      # flatten the jagged arrays to a normal numpy array, turn the whole dictionary to pandas dataframe
      self._branches = pd.DataFrame.from_dict({branch: array.flatten() for branch, array in self._branches.items()})
      #self._branches = awkward.topandas(self._branches, flatten=True)


      # add additional branches
      #self._branches['BToKEE_l_xy_sig'] = self._branches['BToKEE_l_xy'] / np.sqrt(self._branches['BToKEE_l_xy_unc'])
      self._branches['BToKEE_l_xy_sig'] = self._branches['BToKEE_l_xy'] / self._branches['BToKEE_l_xy_unc']
      self._branches['BToKEE_l1_dxy_sig'] = self._branches['BToKEE_l1_dxy'] / self._branches['BToKEE_l1_dxyErr']
      self._branches['BToKEE_l2_dxy_sig'] = self._branches['BToKEE_l2_dxy'] / self._branches['BToKEE_l2_dxyErr']
      self._branches['BToKEE_fit_l1_normpt'] = self._branches['BToKEE_fit_l1_pt'] / self._branches['BToKEE_fit_mass']
      self._branches['BToKEE_fit_l2_normpt'] = self._branches['BToKEE_fit_l2_pt'] / self._branches['BToKEE_fit_mass']
      self._branches['BToKEE_fit_k_normpt'] = self._branches['BToKEE_fit_k_pt'] / self._branches['BToKEE_fit_mass']
      self._branches['BToKEE_fit_normpt'] = self._branches['BToKEE_fit_pt'] / self._branches['BToKEE_fit_mass']

      # general selection
      
      sv_selection = (self._branches['BToKEE_fit_pt'] > 3.0) & (self._branches['BToKEE_l_xy_sig'] > 6.0 ) & (self._branches['BToKEE_svprob'] > 0.01) & (self._branches['BToKEE_fit_cos2D'] > 0.9)
      l1_selection = (self._branches['BToKEE_l1_convVeto']) & (self._branches['BToKEE_fit_l1_pt'] > 1.5) #& (self._branches['BToKEE_l1_mvaId'] > 3.94) #& (np.logical_not(self._branches['BToKEE_l1_isPFoverlap']))
      l2_selection = (self._branches['BToKEE_l2_convVeto']) & (self._branches['BToKEE_fit_l2_pt'] > 0.5) #& (self._branches['BToKEE_l2_mvaId'] > 3.94) #& (np.logical_not(self._branches['BToKEE_l2_isPFoverlap']))
      k_selection = (self._branches['BToKEE_fit_k_pt'] > 0.5) #& (self._branches['BToKEE_k_DCASig'] > 2.0)
      #additional_selection = (self._branches['BToKEE_fit_mass'] > B_LOW) & (self._branches['BToKEE_fit_mass'] < B_UP)

      b_lowsb_selection = (self._branches['BToKEE_fit_mass'] > B_LOWSB_LOW) & (self._branches['BToKEE_fit_mass'] < B_LOWSB_UP)
      b_upsb_selection = (self._branches['BToKEE_fit_mass'] > B_UPSB_LOW) & (self._branches['BToKEE_fit_mass'] < B_UPSB_UP)
      b_sb_selection = b_lowsb_selection | b_upsb_selection
      if self._isMC:
        mc_matched_selection = (self._branches['BToKEE_l1_genPartIdx'] > -0.5) & (self._branches['BToKEE_l2_genPartIdx'] > -0.5) & (self._branches['BToKEE_k_genPartIdx'] > -0.5)
        # B->K J/psi(ee)
        #mc_parent_selection = (abs(self._branches['BToKEE_l1_genMotherPdgId']) == 443) & (abs(self._branches['BToKEE_k_genMotherPdgId']) == 521)
        #mc_chain_selection = (self._branches['BToKEE_l1_genMotherPdgId'] == self._branches['BToKEE_l2_genMotherPdgId']) & (self._branches['BToKEE_k_genMotherPdgId'] == self._branches['BToKEE_l1Mother_genMotherPdgId']) & (self._branches['BToKEE_k_genMotherPdgId'] == self._branches['BToKEE_l2Mother_genMotherPdgId'])

        # B->K*(K pi) J/psi(ee)
        mc_parent_selection = (abs(self._branches['BToKEE_l1_genMotherPdgId']) == 443) & (abs(self._branches['BToKEE_k_genMotherPdgId']) == 313)
        mc_chain_selection = (self._branches['BToKEE_l1_genMotherPdgId'] == self._branches['BToKEE_l2_genMotherPdgId'])
        mc_selection = mc_matched_selection & mc_parent_selection & mc_chain_selection

      #additional_selection = b_sb_selection
      if self._isMC:
        selection = l1_selection & l2_selection & k_selection & mc_selection

      else:
        selection = l1_selection & l2_selection & k_selection

      self._branches = self._branches[selection]

      # fill output
      self.fill_output()

    self.finish()
    print('[BToKLLAnalyzer::run] INFO: Finished')
    self.print_timestamp()


