import os
import sys
from ROOT import *
gROOT.SetBatch(True)
gStyle.SetOptStat(0)
gStyle.SetOptTitle(0)

plots = [
  "BToPhiMuMu_chi2",
  "BToPhiMuMu_eta",
  "BToPhiMuMu_fit_cos2D",
  "BToPhiMuMu_fit_eta",
  "BToPhiMuMu_fit_mass",
  "BToPhiMuMu_fit_phi",
  "BToPhiMuMu_fit_pt",
  "BToPhiMuMu_l_xy",
  "BToPhiMuMu_lep1eta_fullfit",
  "BToPhiMuMu_lep1phi_fullfit",
  "BToPhiMuMu_lep1pt_fullfit",
  "BToPhiMuMu_lep2eta_fullfit",
  "BToPhiMuMu_lep2phi_fullfit",
  "BToPhiMuMu_lep2pt_fullfit",
  "BToPhiMuMu_mass",
  "BToPhiMuMu_mll_fullfit",
  "BToPhiMuMu_mll_llfit",
  "BToPhiMuMu_mll_raw",
  "BToPhiMuMu_mphi_fullfit",
  "BToPhiMuMu_phi",
  "BToPhiMuMu_etaphi_fullfit",
  "BToPhiMuMu_phiphi_fullfit",
  "BToPhiMuMu_pt",
  "BToPhiMuMu_ptphi_fullfit",
  "BToPhiMuMu_svprob",
  "BToPhiMuMu_charge",
]

xtitles = {
  "BToPhiMuMu_chi2":"#chi^{2}",
  "BToPhiMuMu_eta":"Prefit B_{s} #eta",
  "BToPhiMuMu_fit_cos2D":"cos2D",
  "BToPhiMuMu_fit_eta":"Fitted B_{s} #eta",
  "BToPhiMuMu_fit_mass":"Fitted B_{s} mass [GeV]",
  "BToPhiMuMu_fit_phi":"Fitted B_{s} #phi",
  "BToPhiMuMu_fit_pt":"Fitted B_{s} p_{T} [GeV]",
  "BToPhiMuMu_l_xy":"L_{xy}",
  "BToPhiMuMu_lep1eta_fullfit":"Fitted #mu_{1} #eta",
  "BToPhiMuMu_lep1phi_fullfit":"Fitted #mu_{1} #phi",
  "BToPhiMuMu_lep1pt_fullfit":"Fitted #mu_{1} p_{T} [GeV]",
  "BToPhiMuMu_lep2eta_fullfit":"Fitted #mu_{2} #eta",
  "BToPhiMuMu_lep2phi_fullfit":"Fitted #mu_{2} #phi",
  "BToPhiMuMu_lep2pt_fullfit":"Fitted #mu_{2} p_{T} [GeV]",
  "BToPhiMuMu_mass":"Prefit B_{s} mass [GeV]",
  "BToPhiMuMu_mll_fullfit":"Fitted B_{s} m_{#mu#mu} [GeV]",
  "BToPhiMuMu_mll_llfit":"Fitted J/#psi m_{#mu#mu} [GeV]",
  "BToPhiMuMu_mll_raw":"Prefit m_{#mu#mu} [GeV]",
  "BToPhiMuMu_mphi_fullfit":"Fitted B_{s} m_{#phi} [GeV]",
  "BToPhiMuMu_phi":"Prefit B_{s} #phi",
  "BToPhiMuMu_etaphi_fullfit":"Fitted B_{s} #eta_{#phi}",
  "BToPhiMuMu_phiphi_fullfit":"Fitted B_{s} #phi_{#phi}",
  "BToPhiMuMu_pt":"Prefit B_{s} p_{T} [GeV]",
  "BToPhiMuMu_ptphi_fullfit":"Fitted B_{s} p_{T,#phi}",
  "BToPhiMuMu_svprob":"SV prob",
  "BToPhiMuMu_charge":"Charge",

}

fin = TFile("Bs_hists.root")
for plot in plots:
  c = TCanvas("c_{}".format(plot), "c_{}".format(plot), 800, 600)
  h = fin.Get(plot)
  h.GetXaxis().SetTitle(xtitles[plot])
  h.Draw()
  c.SaveAs("plots/{}.pdf".format(plot))
  c.SaveAs("plots/{}.png".format(plot))

