import ROOT
import pandas as pd
import os
import multiprocessing as mp
import sys
sys.path.append('.')
from Bu2KJpsi2KMuMuAnalyzer import Bu2KJpsi2KMuMuAnalyzer


ROOT.gROOT.SetBatch()

def exec_me(command, dryRun=False):
    print(command)
    if not dryRun:
        os.system(command)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    n = max(1, n)
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def analyze(inputfile, output_file, hist=False, isMC=False):
    analyzer = Bu2KJpsi2KMuMuAnalyzer(inputfile, output_file, isMC)
    analyzer.run()

def analyzeParallel(enumfChunk):
    ich, fChunk = enumfChunk
    print("Processing chunk number %i"%(ich))
    output_file = outpath+'/'+args.output_file.replace('.root','').replace('.h5','')+'_subset'+str(ich)+'.root'
    analyze(fChunk, output_file, args.hist, args.isMC)

skim_directory = "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory"
in_txt = {
    "probefilter": "{}/files_BuToKJpsi_ToMuMu_probefilter_SoftQCDnonD.txt".format(skim_directory),
    "inclusive": "{}/files_BuToJpsiK_SoftQCDnonD.txt".format(skim_directory),
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Bs analyzer")
    parser.add_argument("-t", "--tag", default="probefilter", help="probefilter or inclusive")
    #parser.add_argument("-o", "--output_file", dest="output_file", default="Bu_hists.root", help="Output file containing plots")
    parser.add_argument("-n", "--nevents", dest="nevents", type=int, default=-1, help="Maximum number events to loop over")
    parser.add_argument("-r", "--runparallel", dest="runparallel", action='store_true', help="Enable parallel run")
    args = parser.parse_args()

    input_files = []
    with open(in_txt[args.tag]) as f:
        for line in f:
            input_files.append(line.strip())

    output_file = "Bu_hists_{}.root".format(args.tag)

    if not args.runparallel:
        analyzer = Bu2KJpsi2KMuMuAnalyzer(input_files, output_file, True)
        analyzer.start()
        analyzer.run()
        analyzer.finish()

    else:
        #outputBase = "/eos/uscms/store/user/klau/BsPhiLL_output/LowPtElectronSculpting"
        #outputFolder = "BsPhiEE_CutBasedEvaluation"
        global outpath
        #outpath  = "%s/%s"%(outputBase,outputFolder)
        outpath = '.'
        if not os.path.exists(outpath):
            exec_me("mkdir -p %s"%(outpath), False)

        with open(args.inputfiles) as filenames:
            fileList = [f.rstrip('\n') for f in filenames]
        group  = 6
        # stplie files in to n(group) of chunks
        fChunks= list(chunks(fileList,group))
        print ("writing %s jobs"%(len(fChunks)))

        pool = mp.Pool(processes = 4)
        input_parallel = list(enumerate(fChunks))
        #print(input_parallel)
        pool.map(analyzeParallel, input_parallel)
        pool.close()
        pool.join()

        output_file = args.output_file.replace('.root','').replace('.h5','')
        #if args.hist:
        exec_me("hadd -k -f %s/%s %s/%s"%(outpath,output_file+'.root',outpath,output_file+'_subset*.root'))
        exec_me("rm %s/%s"%(outpath,output_file+'_subset*.root'))
        
        '''
        else:
          if os.path.isfile('{}/{}.h5'.format(outpath, output_file)): os.system('rm {}/{}.h5'.format(outpath, output_file))
          #allHDF = [pd.read_hdf(f, 'branches')  for f in ['{}/{}'.format(outpath, output_file+'_subset{}.h5'.format(i)) for i in range(len(fChunks))]]
          allHDF = []
          for f in ['{}/{}'.format(outpath, output_file+'_subset{}.h5'.format(i)) for i in range(len(fChunks))]:
            try:
              allHDF.append(pd.read_hdf(f))
            except ValueError:
              print('Empty HDF file')
          if len(allHDF) != 0:
            outputHDF = pd.concat(allHDF, ignore_index=True)
          else:
            outputHDF = pd.DataFrame()
          outputHDF.to_hdf('{}/{}.h5'.format(outpath, output_file), 'branches', mode='a', format='table', append=True)
          exec_me("rm %s/%s"%(outpath,output_file+'_subset*.h5'))
        '''




