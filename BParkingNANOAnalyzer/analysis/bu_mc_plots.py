import os
import sys
from ROOT import *
gROOT.SetBatch(True)
gStyle.SetOptStat(0)
gStyle.SetOptTitle(0)

plots = [
  "BToKMuMu_chi2",
  "BToKMuMu_eta",
  "BToKMuMu_fit_cos2D",
  "BToKMuMu_fit_eta",
  "BToKMuMu_fit_mass",
  "BToKMuMu_fit_phi",
  "BToKMuMu_fit_pt",
  "BToKMuMu_l_xy",
  "BToKMuMu_fit_l1_eta",
  "BToKMuMu_fit_l1_phi",
  "BToKMuMu_fit_l1_pt",
  "BToKMuMu_fit_l2_eta",
  "BToKMuMu_fit_l2_phi",
  "BToKMuMu_fit_l2_pt",
  "BToKMuMu_mass",
  "BToKMuMu_mll_fullfit",
  "BToKMuMu_mll_llfit",
  "BToKMuMu_mll_raw",
  "BToKMuMu_phi",
  "BToKMuMu_pt",
  "BToKMuMu_svprob",
  "BToKMuMu_charge",
  "BToKMuMu_fit_l_minpt",
]

xtitles = {
  "BToKMuMu_chi2":"#chi^{2}",
  "BToKMuMu_eta":"Prefit B_{u} #eta",
  "BToKMuMu_fit_cos2D":"cos2D",
  "BToKMuMu_fit_eta":"Fitted B_{u} #eta",
  "BToKMuMu_fit_mass":"Fitted B_{u} mass [GeV]",
  "BToKMuMu_fit_phi":"Fitted B_{u} #phi",
  "BToKMuMu_fit_pt":"Fitted B_{u} p_{T} [GeV]",
  "BToKMuMu_l_xy":"L_{xy}",
  "BToKMuMu_fit_l1_eta":"Fitted #mu_{1} #eta",
  "BToKMuMu_fit_l1_phi":"Fitted #mu_{1} #phi",
  "BToKMuMu_fit_l1_pt":"Fitted #mu_{1} p_{T} [GeV]",
  "BToKMuMu_fit_l2_eta":"Fitted #mu_{2} #eta",
  "BToKMuMu_fit_l2_phi":"Fitted #mu_{2} #phi",
  "BToKMuMu_fit_l2_pt":"Fitted #mu_{2} p_{T} [GeV]",
  "BToKMuMu_fit_l_minpt":"Fitted min #mu p_{T} [GeV]", 
  "BToKMuMu_mass":"Prefit B_{u} mass [GeV]",
  "BToKMuMu_mll_fullfit":"Fitted B_{u} m_{#mu#mu} [GeV]",
  "BToKMuMu_mll_llfit":"Fitted J/#psi m_{#mu#mu} [GeV]",
  "BToKMuMu_mll_raw":"Prefit m_{#mu#mu} [GeV]",
  "BToKMuMu_phi":"Prefit B_{u} #phi",
  "BToKMuMu_pt":"Prefit B_{u} p_{T} [GeV]",
  "BToKMuMu_svprob":"SV prob",
  "BToKMuMu_charge":"Charge",

}

fin = TFile("Bu_hists.root")
for tag_type in ["inclusive", "triggered", "tag", "probe"]:
  for plot in plots:
    c = TCanvas("c_{}_{}".format(tag_type, plot), "c_{}".format(plot), 800, 600)
    print("{}_{}".format(tag_type, plot))
    h = fin.Get("{}_{}".format(tag_type, plot))
    h.GetXaxis().SetTitle(xtitles[plot])
    h.Draw()
    c.SaveAs("plots/BuToKMuMu/{}.pdf".format(c.GetName()))
    c.SaveAs("plots/BuToKMuMu/{}.png".format(c.GetName()))

  # Tag-side trigger efficiency
  tagtrigeff = fin.Get("tag_BToKMuMu_fit_pt").Clone()
  tagtrigeff.Divide(fin.Get("tag_BToKMuMu_fit_pt"), fin.Get("triggered_BToKMuMu_fit_pt"), 1, 1, "B")
  c_tagtrigeff = TCanvas("c_trigefftag_BToKMuMu_fit_pt", "HLT_Mu7_IP4", 800, 600)
  tagtrigeff.GetXaxis().SetTitle("p_{T} [GeV]")
  tagtrigeff.GetYaxis().SetTitle("Trigger efficiency")
  tagtrigeff.SetMarkerStyle(21)
  tagtrigeff.Draw()
  c_tagtrigeff.SaveAs("plots/BuToKMuMu/{}.pdf".format(c_tagtrigeff.GetName()))
  c_tagtrigeff.SaveAs("plots/BuToKMuMu/{}.png".format(c_tagtrigeff.GetName()))

  probetrigeff = fin.Get("probe_BToKMuMu_fit_pt").Clone()
  probetrigeff.Divide(fin.Get("probe_BToKMuMu_fit_pt"), fin.Get("triggered_BToKMuMu_fit_pt"), 1, 1, "B")
  c_probetrigeff = TCanvas("c_trigeffprobe_BToKMuMu_fit_pt", "HLT_Mu7_IP4", 800, 600)
  probetrigeff.GetXaxis().SetTitle("p_{T} [GeV]")
  probetrigeff.GetYaxis().SetTitle("Trigger efficiency")
  probetrigeff.SetMarkerStyle(21)
  probetrigeff.Draw()
  c_probetrigeff.SaveAs("plots/BuToKMuMu/{}.pdf".format(c_probetrigeff.GetName()))
  c_probetrigeff.SaveAs("plots/BuToKMuMu/{}.png".format(c_probetrigeff.GetName()))

