#include <iostream>
#include <string>
#include <sstream>
#include <math.h>
#include <algorithm>
#include <numeric>
//ROOT include files
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>

//Fastjet headers
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/tools/MassDropTagger.hh"
#include "fastjet/tools/Filter.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include <fastjet/tools/JHTopTagger.hh>
#include <fastjet/Selector.hh>

//Boost headers
#include <boost/algorithm/string.hpp>
#include <boost/tuple/tuple.hpp>

//custom headers
#include "TopHist.h"
#include "complex_d.h"

using namespace std;
using namespace fastjet;
using namespace pdf;

//----------------------------------------------------------------------
// Some four-vector operators
//----------------------------------------------------------------------
double dot(fastjet::PseudoJet p1, fastjet::PseudoJet p2);
double deltaR(fastjet::PseudoJet p1, fastjet::PseudoJet p2);

/* jet to lepton mistag */
double Pjet_to_lepton(double pt);

/* jet to photon mistag */
double Pjet_to_photon(double pt);

//----------------------------------------------------------------------// forward declaration for printing out info about a jet
//----------------------------------------------------------------------
ostream & operator<<(ostream &, const PseudoJet &);

//----------------------------------------------------------------------
// command line parameters
//----------------------------------------------------------------------
char* getCmdOption(char ** begin, char ** end, const std::string & option);
bool cmdOptionExists(char** begin, char** end, const std::string& option);

//----------------------------------------------------------------------
// Analysis functions
//----------------------------------------------------------------------

// smearing of jets, leptons and photons. 
fastjet::PseudoJet smear_jet(fastjet::PseudoJet jet_in);
fastjet::PseudoJet smear_lepton(fastjet::PseudoJet lepton_in, int lepton_id);
fastjet::PseudoJet smear_photon(fastjet::PseudoJet photon_in);

// acceptance efficiency for leptons, photons, jets
bool lepton_efficiency_accept(fastjet::PseudoJet lepton_in, int lepton_id);
bool photon_efficiency_accept(fastjet::PseudoJet photon_in);
bool jet_efficiency_accept(fastjet::PseudoJet jet_in);
bool btag_hadrons(fastjet::PseudoJet jet);


//analysis functions
double analyze_event(fastjet::PseudoJet photon1, fastjet::PseudoJet photon2, fastjet::PseudoJet cjet, fastjet::PseudoJet bjet, fastjet::PseudoJet lepton, fastjet::PseudoJet etmiss, double evweight_i);
double Pb_to_b(double pt);
double Pb_to_c(double pt);
double Pc_to_b(double pt);
double Pjet_to_b(double pt);
// jet to lepton mistag prob
double Pjet_to_photon(double pt);
double btag_weight(fastjet::PseudoJet jet, bool btag, bool ctag);
double atag_weight(fastjet::PseudoJet jet, bool btag, bool ctag);
double ctag_weight(fastjet::PseudoJet jet, bool btag, bool ctag);
std::pair<fastjet::PseudoJet, fastjet::PseudoJet> get_Wvectors(fastjet::PseudoJet plepton, fastjet::PseudoJet pmiss);
std::pair<fastjet::PseudoJet, fastjet::PseudoJet> get_Wvectors_GTX(fastjet::PseudoJet plepton, fastjet::PseudoJet pmiss);
std::pair<complex_ld, complex_ld> pnuz_fromw(fastjet::PseudoJet plepton, fastjet::PseudoJet pmiss);
std::pair<complex_d, complex_d> quadsolve(complex_ld a, complex_ld b, complex_ld c);

//get all pairings:
void generatePairings(int* items, int itemcount, int start);

//check if integer is in vector of integers
bool is_in(int a, vector<int> intvec);

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

// IDs of B-hadrons used by the btag_hadrons function
int bhadronid[105] = {5122, -5122, 15122, -15122, 5124, -5124, 5334, -5334, 5114, -5114, 5214, -5214, 5224, -5224, 5112, -5112, 5212, -5212, 5222, -5222, 15322, -15322, 15312, -15312, 15324, -15324, 15314, -15314, 5314, -5314, 5324, -5324, 5132, -5132, 5232, -5232, 5312, -5312, 5322, -5322, 551, 10555, 100551, 200551, 553, 557, 555, 100555, 200555, 20523, -20523, 20513, -20513, 20543, -20543, 20533, -20533, 511, 521, -511, -521, 531, -531, 541, -541, 513, 523, -513, -523, 533, -533, 543, -543, 10513, 10523, -10513, -10523, 10533, -10533, 10543, -10543, 10511, 10521, -10511, -10521, 10531, -10531, 10541, -10541, 20513, 20523, -20513, -20523, 20533, -20533, 20543, -20543, 515, 525, -515, -525, 535, -535, 545, -545};

//particle masses
double mw = 80.4;
double mtop = 173.0;
double mhiggs = 125.0;

/*
 * CREATE ROOT CHAIN TO READ IN THE FILES
 */
TChain t("Data");


/* 
 * DECLARE RANDOM NUMBERS
 */ 
TRandom3 rnd;
TRandom3 rndint;

/***** 
 ***** SWITCHES FOR SMEARING/EFFICIENCIES
 *****/
bool donotsmear_jets = 1;
bool donotsmear_leptons = 1;
bool donot_apply_efficiency = 1;
bool donotsmear_photons = 1;

/* 
 * cut counters
 */ 
double passed_lepton_cuts(0.);   


/* 
 * CUTS DEFINED HERE IN GEV
*/
 
/* JET CUTS */

double cut_pt_jet(25.0); //pt cut for jets
double cut_eta_jet(4.0); //pseudo-rapidity cut for jets

/* B-jet CUTS */
double cut_pt_bjet(35.0); //pt cut for b-jets
double cut_eta_bjet(3.3); //pseudo-rapidity cut for b-jets
double cut_dRbbmin(0.3); //minimum delta R between b-jets
double cut_pt_bjet1(170.0); //pt cut for b-jets
double cut_pt_bjet2(135.0); //pt cut for b-jets
double cut_pt_bjet3(35.0); //pt cut for b-jets



/* RECO Higgs CUTS */ 
double cut_pt_higgs1(200); //minimum pt for reco higgs 1
double cut_pt_higgs2(190); //minimum pt for reco higgs 2
double cut_pt_higgs3(20.0); //minimum pt for reco higgs 3
double cut_chisq_min(26.); //maximum of the minimum (best) chi-squared
double cut_DeltaM_min(8); //maximum DeltaM_min
double cut_DeltaM_med(8); //maximum DeltaM_med
double cut_DeltaM_max(8); //maximum DeltaM_max
double cut_dR_higgses(3.5); //maximum delta R beteween reco Higgses
double cut_dR_hbbreco(3.5); //maximum delta R between b-jets in a reco Higgs

bool perfect_tagging = true;

std::vector<std::vector<int>> pairs_of_six;
  
int main(int argc, char *argv[]) {

  //take command line options
  char* output;
  char* infile = "";
  if(argv[1]) { infile = argv[1]; } else { cout << "Use: ./HwSimAnalysis [input] [options]" << endl; exit(1); }

  //set the variables and addresses to be read from root file
  //total number of particles in an event
  int numparticles;
  //total number of jets in an event
  int numJets;
  //total number of b-jets in an event
  int numbJets;
  //total number of photons in an event
  int numPhotons;
  //total number of Electrons in an event
  int numElectrons;
  //total number of Positrons in an event
  int numPositrons;
  //total number of Muons in an event
  int numMuons;
  //total number of Muons in an event
  int numantiMuons;
  /** particle information in the order: 
   * 4 momenta (E,x,y,z), id, other info
   **/
  double objects[8][1000];
  /* the event weight */ 
  double evweight;
  /* The vector of jets */
  double theJets[4][100];
  /* The vector of jets */
  double thebJets[4][100];
  /* The vector of photons */
  double thePhotons[4][100];
  /* The vector of muons */
  double theMuons[4][100];
  /* The vector of electrons */
  double theElectrons[4][100];
   /* The vector of anti-muons */
  double theantiMuons[4][100];
  /* The vector of positrons */
  double thePositrons[4][100];
  /* The missing energy four-vector */
  double theETmiss[4];
  /* c-Tagging containers corresponding to the jets */
  double cTag[100];


  /* 
   * SET THE ROOT BRANCH ADDRESSES
   */
  //comment out numparticles and objects if not saving them
  //t.SetBranchAddress("numparticles",&numparticles);
  //t.SetBranchAddress("objects",&objects);
  
  t.SetBranchAddress("evweight",&evweight);
  
  t.SetBranchAddress("theJets", &theJets);
  t.SetBranchAddress("numJets", &numJets);
  t.SetBranchAddress("cTag", &cTag);

  t.SetBranchAddress("thebJets", &thebJets);
  t.SetBranchAddress("numbJets", &numbJets);
        
  t.SetBranchAddress("thePhotons", &thePhotons);
  t.SetBranchAddress("numPhotons", &numPhotons);
        
  t.SetBranchAddress("theMuons", &theMuons);
  t.SetBranchAddress("numMuons", &numMuons);
        
  t.SetBranchAddress("theantiMuons", &theantiMuons);
  t.SetBranchAddress("numantiMuons", &numantiMuons);
  
  t.SetBranchAddress("thePositrons", &thePositrons);
  t.SetBranchAddress("numPositrons", &numPositrons);
        
  t.SetBranchAddress("theElectrons", &theElectrons);
  t.SetBranchAddress("numElectrons", &numElectrons);
        
  t.SetBranchAddress("theETmiss", &theETmiss);


  /* Set up random number
   * generator
   */ 
  rnd.SetSeed(14101983);


  /* Add up all the input 
   * files to the chain
   */ 
  string stringin = "";
  ifstream inputlist;
  if (std::string(infile).find(".input") != std::string::npos) {
    inputlist.open(infile);
    if(!inputlist) {  cerr << "Error: Failed to open input file " << infile << endl; exit(1); }
    while(inputlist) { 
      inputlist >> stringin; 
      if(stringin!="") { t.Add(stringin.c_str()); 
        cout << "Adding " << stringin.c_str() << endl;
      }
      stringin = "";
    }
    inputlist.close();
  } else if (std::string(infile).find(".root") != std::string::npos) {
    cout << "Adding " << infile << endl;
    t.Add(infile);
  }

  /* Get Number of events
   * and print
   */
  int EventNumber(int(t.GetEntries()));
  cout << "Total number of events in " << infile << " : " << EventNumber << endl;

  // warn if the perfect tagging flag is on
  if(perfect_tagging) cout << "WARNING: Perfect tagging of b-jets and c-jets and no mis-tagging of them enabled" << endl;

  /* 
   * -b: USED TO REANALYZE PREVIOUSLY PASSED EVENTS ONLY, DEFAULT IS ALL EVENTS 
   */
  
  //whether the analysis performed is level-2 or level-3 
  bool basic = true;
  if(cmdOptionExists(argv, argv+argc, "-b")) {
    cout << "Looking for .evp2 file, running over all events" << endl;
    basic = false;
  }


  /* 
   * -t: ADD AN EXTENSIN TAG TO YOUR OUTPUT FILES
   */
  string tag;
  tag = "";
  if(cmdOptionExists(argv, argv+argc, "-t")) {
    tag = getCmdOption(argv, argv + argc, "-t");
    tag = "-" + tag;
    cout << "Adding tag: " << tag << endl;
  }

  /* 
   * -n: RUN FROM START OF FILE UP TO A GIVEN NUMBER OF EVENTS
   */
  char * switch_maxevents;
  char * switch_minevents;
  int maxevents(0), minevents(0);
  if(cmdOptionExists(argv, argv+argc, "-n")) {
    switch_maxevents = getCmdOption(argv, argv + argc, "-n");  
    maxevents=(atoi(switch_maxevents));	       
    if(maxevents > EventNumber) { maxevents = EventNumber; } 
    cout << "Analyzing up to " << maxevents << endl;
    if(maxevents < 1 || maxevents > 1E10) { cout << "Error: maxevents must be in the range [1,1E10]" << endl; exit(1); } 
  }
  
  /* 
   * -nmax: RUN FROM START OF FILE UP TO A GIVEN NUMBER OF EVENTS, TO BE USED IN CONJUNCTION WITH -nmin
   */
  //maximum number of events to analyze
  if(cmdOptionExists(argv, argv+argc, "-nmax") && !cmdOptionExists(argv, argv+argc, "-n")) {
   switch_maxevents = getCmdOption(argv, argv + argc, "-nmax");  
    maxevents=(atoi(switch_maxevents));	       
    if(maxevents > EventNumber) { maxevents = EventNumber; } 
    cout << "Analyzing up to " << maxevents << endl;
    if(maxevents < 1 || maxevents > 1E10) { cout << "Error: maxevents must be in the range [1,1E10]" << endl; exit(1); } 
  } 
  if(!cmdOptionExists(argv, argv+argc, "-nmax") && !cmdOptionExists(argv, argv+argc, "-n")) { maxevents = EventNumber; }

  /* 
   * -nmin: RUN FROM POINT nmin OF FILE UP TO A GIVEN NUMBER OF EVENTS SPECIFIED BY -nmax
   */
  //starting number of events to analyse
  if(cmdOptionExists(argv, argv+argc, "-nmin")) {
    switch_minevents = getCmdOption(argv, argv + argc, "-nmin");  
    minevents=(atoi(switch_minevents));	       
    if(minevents > maxevents) { minevents = 0; }
    cout << "Analyzing from " << minevents << endl;
    if(minevents < 1 || minevents > 1E10) { cout << "Error: minevents must be in the range [1,1E10]" << endl; exit(1); } 
  }


  /* 
   * CREATE THE OUTPUT FILE STRINGS 
   */
  string outnew = "";
  outnew = std::string(infile);
  string replacement = tag + ".top";
  boost::replace_all(outnew, ".root", replacement);
  boost::replace_all(outnew, ".input", replacement);        
  char* output2 = new char[outnew.length() + 1];
  //  cout << outnew.c_str() << endl;
  strcpy (output2, outnew.c_str());
  output = output2;
  
  char* output_dat;
  string outnew2 = "";
  outnew2 = std::string(infile);
  replacement = tag + ".smear.dat";
  boost::replace_all(outnew2, ".root", replacement);
  boost::replace_all(outnew2, ".input", replacement);        
  char* output3 = new char[outnew2.length() + 1];
  strcpy (output3, outnew2.c_str());
  output_dat = output3;
  ofstream outdat(output_dat, ios::out);

  //load events that have passed the second stage of analysis
  //if basic = false;
  string ineventpass;
  ifstream inevt;
  string inevt_curr;
  int passed_event[20000];  
  int npassed_previous(0);
  if(basic == false) { 
    ineventpass = std::string(infile);
    replacement = tag + ".evp";

    boost::replace_all(ineventpass,".input", replacement);
    boost::replace_all(ineventpass,".root", replacement);
    inevt.open(ineventpass.c_str());
    if(!inevt) { cerr << "Error: Cannot open "<< ineventpass.c_str() << endl; exit(1); } 
    for(int ii = 0; ii < 1000; ii++) { passed_event[ii] = -1; }
    while(inevt) { 
      inevt >> inevt_curr;
      // cout << inevt_curr.c_str() << endl;
      passed_event[npassed_previous] = atoi(inevt_curr.c_str());
      npassed_previous++;
    }
  }
  //for(int pp = 0; pp < npassed_previous; pp++) { coust << passed_event[pp] << endl; }
 
  string outeventpass = ""; 
  ofstream outevp;

  if(basic == false) { 
    outeventpass = std::string(infile);
    replacement = tag + ".evp2";
    boost::replace_all(outeventpass,".root", replacement);
    boost::replace_all(outeventpass,".input", replacement);
    boost::replace_all(outeventpass,".top", replacement);
    outevp.open(outeventpass.c_str());
  } else if(basic == true) {
    outeventpass = std::string(infile);
    replacement = tag + ".evp";
    boost::replace_all(outeventpass,".root", replacement);
    boost::replace_all(outeventpass,".input", replacement);
    boost::replace_all(outeventpass,".top", replacement);
    outevp.open(outeventpass.c_str());
  }

  /*
   * PREPARES THE OUTPUT ARRAY FOR *_var.root: USED FOR FURTHER ANALYSIS
   */
  std::cout << "Preparing Root Tree for event variables" << endl;
  TTree* Data2;
  TFile* dat2;
  string fnameroot = std::string(infile);
  replacement = tag + "_var.smear.root";
  boost::replace_all(fnameroot,".root", replacement);
  boost::replace_all(fnameroot,".input", replacement);
  dat2 = new TFile(fnameroot.c_str(), "RECREATE");
  Data2 = new TTree ("Data2", "Data Tree");
  //variables to fill in the .root file
  double variables[21]; 
  double weight;
  Data2->Branch("variables", &variables, "variables[21]/D");
  Data2->Branch("weight", &weight);

 
  /* 
   * COUNTERS FOR NUMBER OF EVENTS THAT PASS CUTS
   */
  double pass_6b(0); //passed reconstruction of 6 b-jets
  double pass_ptb(0); //passed minimum pt of all 6 b-jets
  double pass_drbb(0); //passed dR > 0.3 for all pairings
  double pass_pthiggses(0); //passed reco higgses pT cuts
  double pass_chisq(0); //passed the chi-squared minimum cut
  double pass_DeltaM(0); //passed delta M cut
  double pass_dRhiggses(0); //passed reco higgses dR cuts
  double pass_dRbbhiggses(0); //passed reco higgses dR(b,b) cuts

  double passcuts(0); //passed all cuts
  double eventcount(0); //counting the events (no weights)
  double total_event_in(0); //counting the events in (no weights)
  double total_weight_in(0);//the total weight before analysis


  /* 
   * PARAMETERS AND SWITCHES
   */
 
  /* 
   * HISTOGRAMS DEFINED HERE 
   */
  TopHist h_dummy(10,output,"dummy histo", 0,1);
  TopHist h_pT_jets(60,output,"pT of all jets",0, 300);
  TopHist h_pT_leptons(60,output,"pT of leptons",0, 300);
  TopHist h_pT_b(60,output,"pT of reco b jets",0, 300);
  TopHist h_pT_b1(60,output,"pT of reco b jet 1",0, 300);
  TopHist h_pT_b2(60,output,"pT of reco b jet 2",0, 300);
  TopHist h_pT_b3(60,output,"pT of reco b jet 3",0, 300);
  TopHist h_pT_b4(60,output,"pT of reco b jet 4",0, 300);
  TopHist h_pT_b5(60,output,"pT of reco b jet 5",0, 300);
  TopHist h_pT_b6(60,output,"pT of reco b jet 6",0, 300);
  TopHist h_DeltaM_min(60,output,"Delta M min",0, 300);
  TopHist h_DeltaM_med(60,output,"Delta M med",0, 300);
  TopHist h_DeltaM_max(60,output,"Delta M max",0, 300);
  TopHist h_pT_h1(60,output,"pT of Higgs 1",0, 300);
  TopHist h_pT_h2(60,output,"pT of Higgs 2",0, 300);
  TopHist h_pT_h3(60,output,"pT of Higgs 3",0, 300);
  TopHist h_pT_dRhh(60,output,"delta R between Higgs bosons",0, 3.14153);
  TopHist h_m6b(100,output,"6b invariant mass",0, 1000);


 
  /*
   *
   * LOOP OVER EVENTS
   * AND
   * PERFORM ANALYSIS
   *
   */
  bool RESET_WEIGHTS_TO_UNITY = false;
  if(RESET_WEIGHTS_TO_UNITY) {
    cout << "WARNING: RESETTING ALL WEIGHTS TO = 1" << endl;
  }

  
  int listind[6] = {0,1,2,3,4,5};
  generatePairings(listind, 6, 0);

  for(int pp = 0; pp < pairs_of_six.size(); pp++) {
    for(int ps = 0; ps < pairs_of_six[pp].size(); ps++) { 
      cout << pairs_of_six[pp][ps] << ",";
    }
    cout << endl;
  }

  bool perform_analysis_on_event = false;
  for(int ii = minevents; ii < maxevents; ii++) {
    
    /* IF LEVEL 3 ANALYSIS THEN
     * CHECK IF EVENT IS IN .evp FILE
     */ 
    perform_analysis_on_event = false;
    if(basic == false) { 
       for(int pp = 0; pp < npassed_previous; pp++) { if(ii == passed_event[pp]) { perform_analysis_on_event = true; } }
    }
    if(!perform_analysis_on_event && basic == false) { continue; }

    /* GRAB EVENT ENTRY
     * FROM ROOT FILE
     * AND PRINT EVENT NUMBER
     */
    t.GetEntry(ii);
    if(RESET_WEIGHTS_TO_UNITY) { 
      evweight = 1.0;
    }


    if(ii%1 == 0) { cout << "Event number: " << ii << "\r" << flush; }

    /*
     * PUSH BACK JETS, LEPTONS & PHOTONS INTO PSEUDOJETS
     */
    
    vector<fastjet::PseudoJet> bJets_unsort, Jets_unsort, Electrons_unsort, Positrons_unsort, Muons_unsort, AntiMuons_unsort, Photons_unsort, Leptons_unsort;

    fastjet::PseudoJet bjetcan, jetcan, photoncan;
    
    for(int jj = 0; jj < numJets; jj++) {
      jetcan = fastjet::PseudoJet(theJets[1][jj], theJets[2][jj], theJets[3][jj], theJets[0][jj]);
      jetcan.set_user_index(cTag[jj]);
      if(jetcan.perp() > cut_pt_jet && fabs(jetcan.eta()) < cut_eta_jet && jet_efficiency_accept(jetcan)) { 
	Jets_unsort.push_back(smear_jet(jetcan));
      }
    }
     for(int jj = 0; jj < numbJets; jj++) {
       bjetcan = fastjet::PseudoJet(thebJets[1][jj], thebJets[2][jj], thebJets[3][jj], thebJets[0][jj]);
       if(bjetcan.perp() > cut_pt_bjet && fabs(bjetcan.eta()) < cut_eta_bjet && jet_efficiency_accept(bjetcan)) { 
	 bJets_unsort.push_back(smear_jet(bjetcan));
       }
     }
  
     for(int jj = 0; jj < numPhotons; jj++) {
       photoncan = fastjet::PseudoJet(thePhotons[1][jj], thePhotons[2][jj], thePhotons[3][jj], thePhotons[0][jj]);
	 Photons_unsort.push_back(smear_photon(photoncan));
     }
     for(int jj = 0; jj < numElectrons; jj++) {
       fastjet::PseudoJet Electron = fastjet::PseudoJet(theElectrons[1][jj], theElectrons[2][jj], theElectrons[3][jj], theElectrons[0][jj]);
       if(lepton_efficiency_accept(Electron, 11)) {
	 Electron.set_user_index(11);
	 Electrons_unsort.push_back(smear_lepton(Electron, 11));
	 Leptons_unsort.push_back(smear_lepton(Electron, 11));
       }
     }
     for(int jj = 0; jj < numPositrons; jj++) {
       fastjet::PseudoJet Positron = fastjet::PseudoJet(thePositrons[1][jj], thePositrons[2][jj], thePositrons[3][jj], thePositrons[0][jj]);
       if(lepton_efficiency_accept(Positron, 11)) {
	 Positron.set_user_index(-11);
	 Positrons_unsort.push_back(smear_lepton(Positron, -11));
	 Leptons_unsort.push_back(smear_lepton(Positron, -11));
       }
     }
     for(int jj = 0; jj < numMuons; jj++) {
       fastjet::PseudoJet Muon = fastjet::PseudoJet(theMuons[1][jj], theMuons[2][jj], theMuons[3][jj], theMuons[0][jj]);
       if(lepton_efficiency_accept(Muon, 13)) {
	 Muon.set_user_index(13);
	 Muons_unsort.push_back(smear_lepton(Muon, 13));
	 Leptons_unsort.push_back(smear_lepton(Muon, 13));
       }
     }
     for(int jj = 0; jj < numantiMuons; jj++) {
       fastjet::PseudoJet antiMuon = fastjet::PseudoJet(theantiMuons[1][jj], theantiMuons[2][jj], theantiMuons[3][jj], theantiMuons[0][jj]);
       if(lepton_efficiency_accept(antiMuon, 13)) { 
	 antiMuon.set_user_index(-13);
	 AntiMuons_unsort.push_back(smear_lepton(antiMuon, -13));
	 Leptons_unsort.push_back(smear_lepton(antiMuon, -13));
       }
     }
     

     /*
      * SORT RECONSTRUCTED OBJETS BY PT
      */
     vector<fastjet::PseudoJet> bJets, Jets, Electrons, Positrons, Muons, AntiMuons, Photons, Leptons, cJets, LightJets;

     Jets = sorted_by_pt(Jets_unsort);
     bJets = sorted_by_pt(bJets_unsort);
     Photons = sorted_by_pt(Photons_unsort);
     numJets = Jets.size();
     numbJets = bJets.size();
     numPhotons = Photons.size();
     
     Electrons = sorted_by_pt(Electrons_unsort);
     Positrons = sorted_by_pt(Positrons_unsort);
     Muons = sorted_by_pt(Muons_unsort);
     AntiMuons = sorted_by_pt(AntiMuons_unsort);
     Leptons = sorted_by_pt(Leptons_unsort);

     numElectrons = Electrons.size();
     numPositrons = Positrons.size();
     numMuons = Muons.size();
     numantiMuons = AntiMuons.size();
     
     int numLeptons = numElectrons + numPositrons + numMuons + numantiMuons;

     //fill in the input weight:
     total_weight_in += evweight;
     total_event_in++;
     
     fastjet::PseudoJet ETmiss = fastjet::PseudoJet(theETmiss[1], theETmiss[2], theETmiss[3], theETmiss[0]);

     /*
      * FILL IN THE HISTOGRAMS
      */
     for(int jj = 0; jj < numJets; jj++) {
       h_pT_jets.thfill(Jets[jj].perp(), evweight);
     }

     for(int jj = 0; jj < numLeptons; jj++) {
       h_pT_leptons.thfill(Leptons[jj].perp(), evweight);
     }

     /* 
      * CUTS START HERE
      */

     //Fill in the number of b-jets and number of photons until we reach two of each
     fastjet::PseudoJet bJet1, bJet2, bJet3, bJet4, bJet5, bJet6;
     
     //select the six highest-pT b-jets
     if(numbJets >= 6) {
       bJet1 = bJets[0];
       bJet2 = bJets[1];
       bJet3 = bJets[2];
       bJet4 = bJets[3];
       bJet5 = bJets[4];
       bJet6 = bJets[5];
       /*cout << "bJets:" << endl;
       cout << bJet1 << endl;
       cout << bJet2 << endl;
       cout << bJet3 << endl;
       cout << bJet4 << endl;
       cout << bJet5 << endl;
       cout << bJet6 << endl;*/

       evweight *= btag_weight(bJets[0], 1, 0) * btag_weight(bJets[1], 1, 0) * btag_weight(bJets[2], 1, 0) * btag_weight(bJets[3], 1, 0) *btag_weight(bJets[4], 1, 0) * btag_weight(bJets[5], 1, 0);
     } else continue;

      //check that the distance between any two b-jets is larger than DeltaR = 0.3
     bool dRbbfail = false;
     for(int b1=0; b1 < 6; b1++) {
       for(int b2=0; b2 < 6; b2++) {
	 if(b1!=b2) { 
	   double dRbb = deltaR(bJets[b1],bJets[b2]);
	   if(dRbb < cut_dRbbmin) dRbbfail = true;
	 }
       }
     }
     if(dRbbfail == true) continue;
     pass_drbb+=evweight;

     pass_6b+=evweight;

     //check that the pT of the b-jets is larger than cut_pt_bjet
     //also check highest 3 b-jet pts
     /*if(bJet1.perp() < cut_pt_bjet1) continue;
     if(bJet2.perp() < cut_pt_bjet2) continue;
     if(bJet3.perp() < cut_pt_bjet3) continue;
     if(bJet4.perp() < cut_pt_bjet3) continue;
     if(bJet5.perp() < cut_pt_bjet3) continue;
     if(bJet6.perp() < cut_pt_bjet3) continue;*/

     pass_ptb+=evweight;

     //loop over the 15 possible pairings of the 6 b-jets and calculate invariant mass for each "higgs boson candiate"
     double chisq_min(1E99);
     int mincombo(-1);
     std::vector<double> mbb; //reco higgs masses for optimal combo
     std::vector<fastjet::PseudoJet> ph; //reco higgses for optimal combo
     for(int pp = 0; pp < pairs_of_six.size(); pp++) {
       fastjet::PseudoJet bb1;
       fastjet::PseudoJet bb2;
       fastjet::PseudoJet bb3;
       bb1 = bJets[pairs_of_six[pp][0]] + bJets[pairs_of_six[pp][1]]; 
       bb2 = bJets[pairs_of_six[pp][2]] + bJets[pairs_of_six[pp][3]];
       bb3 = bJets[pairs_of_six[pp][4]] + bJets[pairs_of_six[pp][5]];
       /*cout << bJets[pairs_of_six[pp][0]] << endl;
       cout << bJets[pairs_of_six[pp][1]] << endl;
       cout << bb1 << endl;*/
       double mbb1 = bb1.m();
       double mbb2 = bb2.m();
       double mbb3 = bb3.m();
       //for(int ps = 0; ps < pairs_of_six[pp].size(); ps++) cout << pairs_of_six[pp][ps] << ",";
       //cout << mbb1 << " " << mbb2 << " " << mbb3 << endl;
       double chisq_combo = sqrt(pow ((mbb1-mhiggs), 2)+ pow( (mbb2-mhiggs), 2) + pow((mbb3-mhiggs),2)); 
       //cout << "\tchisq = " << chisq_combo << endl;
       if(chisq_combo < chisq_min) { mbb.clear(); chisq_min = chisq_combo; mincombo = pp; mbb.push_back(fabs((mbb1-mhiggs))); mbb.push_back(fabs((mbb2-mhiggs))); mbb.push_back(fabs((mbb3-mhiggs))); ph.push_back(bb1); ph.push_back(bb2); ph.push_back(bb3); }
     }
     //cout << "min. chi-sq = " << chisq_min << " for combo " << mincombo << endl;
     std::vector<double> DeltaM_unsort; //store the differences of the optimal combination with the Higgs mass in ascending order (DeltaM_min, DeltaM_med, DeltaM_max)
     for(int m = 0; m < 3; m++) { DeltaM_unsort.push_back(mbb[m]); /*cout << "mbb = " << mbb[m] << endl;*/ }

     //DeltaM_index contains the indices of DeltaM_unsort in assending order
     //so: DeltaM_unsort[DeltaM_index[0]] is DeltaM_min, 1 is DeltaM_med and 2 is DeltaM_max
     std::vector<int> DeltaM_index(DeltaM_unsort.size());
     std::size_t n(0);
     std::generate(std::begin(DeltaM_index), std::end(DeltaM_index), [&]{ return n++; });
     std::sort(  std::begin(DeltaM_index), std::end(DeltaM_index), [&](int i1, int i2) { return DeltaM_unsort[i1] < DeltaM_unsort[i2]; } );
     ph = sorted_by_pt(ph);

     //impose further cuts:



     //chi-sq cut:
     //if(chisq_min > cut_chisq_min) continue;
     pass_chisq+=evweight;
     
     //DeltaM cuts:
     cout << "DeltaM:" << endl;
     cout << DeltaM_unsort[DeltaM_index[0]] << "\t" << DeltaM_unsort[DeltaM_index[1]] << "\t" << DeltaM_unsort[DeltaM_index[2]] << endl;
     //if(DeltaM_unsort[DeltaM_index[0]] > cut_DeltaM_min || DeltaM_unsort[DeltaM_index[1]] > cut_DeltaM_med || DeltaM_unsort[DeltaM_index[2]] > cut_DeltaM_max) continue;
     pass_DeltaM+=evweight;

     
     //pT of reconstructed Higgs bosons:
     //if(ph[0].perp() < cut_pt_higgs1 || ph[1].perp() < cut_pt_higgs2 || ph[2].perp() < cut_pt_higgs3) continue;
     pass_pthiggses+=evweight;

     //delta R(bb) between b's in reco higgses:
     //if(deltaR(bJets[pairs_of_six[mincombo][0]], bJets[pairs_of_six[mincombo][1]]) > cut_dR_hbbreco || deltaR(bJets[pairs_of_six[mincombo][2]], bJets[pairs_of_six[mincombo][3]]) > cut_dR_hbbreco | deltaR(bJets[pairs_of_six[mincombo][4]], bJets[pairs_of_six[mincombo][5]]) > cut_dR_hbbreco) continue;
     pass_dRbbhiggses+=evweight;

          //delta R between Higgs bosons:
     //if(deltaR(ph[0], ph[1]) > cut_dR_higgses || deltaR(ph[0], ph[2]) > cut_dR_higgses || deltaR(ph[1], ph[2]) > cut_dR_higgses) continue;
     pass_dRhiggses+=evweight;





     /*
     * DOES THE EVENT PASS ALL THE CUTS?
     * IF SO INCREMENT THE WEIGHT
     */ 
     passcuts+=evweight;
     eventcount++;


     
     
     /* 
      * calculate variables for the _var.root file and plot:
      */
     double m6b = (bJets[0]+bJets[1]+bJets[2]+bJets[3]+bJets[4]+bJets[5]).m();
     
     /*
      * Fill in the _var.root file for further analysis.
      */
     variables[0] = evweight;
     variables[1] = bJet1.perp();
     variables[2] = bJet2.perp();
     variables[3] = bJet3.perp();
     variables[4] = bJet4.perp();
     variables[5] = bJet5.perp();
     variables[6] = bJet6.perp();
     variables[7] = m6b;
     variables[8] = chisq_min;
     variables[9] = DeltaM_unsort[DeltaM_index[0]];
     variables[10] = DeltaM_unsort[DeltaM_index[1]];
     variables[11] = DeltaM_unsort[DeltaM_index[2]];
     variables[12] = ph[0].perp();
     variables[13] = ph[1].perp();
     variables[14] = ph[2].perp();
     variables[15] = deltaR(ph[0], ph[1]);
     variables[16] = deltaR(ph[0], ph[2]);
     variables[17] = deltaR(ph[1], ph[2]);
     variables[18] = deltaR(bJets[pairs_of_six[mincombo][0]], bJets[pairs_of_six[mincombo][1]]);
     variables[19] = deltaR(bJets[pairs_of_six[mincombo][2]], bJets[pairs_of_six[mincombo][3]]);
     variables[20] = deltaR(bJets[pairs_of_six[mincombo][4]], bJets[pairs_of_six[mincombo][5]]);
       

     weight = evweight;
     
     Data2->Fill();

     
     /* fill in Histograms: */
     h_pT_b.thfill(bJets[0].perp());
     h_pT_b.thfill(bJets[1].perp());
     h_pT_b.thfill(bJets[2].perp());
     h_pT_b.thfill(bJets[3].perp());
     h_pT_b.thfill(bJets[4].perp());
     h_pT_b.thfill(bJets[5].perp());
     h_pT_b1.thfill(bJets[0].perp());
     h_pT_b2.thfill(bJets[1].perp());
     h_pT_b3.thfill(bJets[2].perp());
     h_pT_b4.thfill(bJets[3].perp());
     h_pT_b5.thfill(bJets[4].perp());
     h_pT_b6.thfill(bJets[5].perp());
     h_DeltaM_min.thfill(DeltaM_unsort[DeltaM_index[0]]);
     h_DeltaM_med.thfill(DeltaM_unsort[DeltaM_index[1]]);
     h_DeltaM_max.thfill(DeltaM_unsort[DeltaM_index[2]]);
     h_pT_h1.thfill(ph[0].perp());
     h_pT_h2.thfill(ph[1].perp());
     h_pT_h3.thfill(ph[2].perp());
     h_pT_dRhh.thfill(deltaR(ph[0], ph[1]));
     h_pT_dRhh.thfill(deltaR(ph[0], ph[2]));
     h_pT_dRhh.thfill(deltaR(ph[1], ph[2]));
     h_m6b.thfill(m6b);
     
     /* IF EVENT HAS PASSED CUTS
      * PRINT TO .evp or .evp2 FILE 
      * INCREMENT AND CONTINUE
      */
     outevp << ii << endl;
     
  } /* LOOP OVER EVENTS ENDS HERE 
     * ENDS HERE
     */
  Data2->GetCurrentFile();
  Data2->Write();
  dat2->Close();
  cout << "A root tree has been written to the file: " << fnameroot << endl;
		
   /* OUTPUT HISTOGRAMS
		 * HERE AND 
		 * FINISH
   */
  h_dummy.plot(output,1,0);
		// "inclusive plot"
  h_pT_jets.add(output,1,0);
  h_pT_leptons.add(output,1,0);
  h_pT_b.add(output,1,0);
  h_pT_b1.add(output,1,0);
  h_pT_b2.add(output,1,0);
  h_pT_b3.add(output,1,0);
  h_pT_b4.add(output,1,0);
  h_pT_b5.add(output,1,0);
  h_pT_b6.add(output,1,0);
  h_DeltaM_min.add(output,1,0);
  h_DeltaM_med.add(output,1,0);
  h_DeltaM_max.add(output,1,0);
  h_pT_h1.add(output,1,0);
  h_pT_h2.add(output,1,0);
  h_pT_h3.add(output,1,0);
  h_pT_dRhh.add(output,1,0);
  h_m6b.add(output,1,0);
  
  cout << "------------------" << endl;
  cout << "total weight in =\t\t\t\t\t\t" <<  total_weight_in << endl;
  cout << "total MC events in =\t\t\t\t\t\t" << total_event_in << endl;
  cout << "------------------" << endl;
  cout << "cuts/counters:" << endl;
  cout << "6bs:\t\t\t\t\t\t\t\t" <<  pass_6b << endl;
  cout << "6bs with pT > [" << cut_pt_bjet1 << ", " << cut_pt_bjet2 << ", " << cut_pt_bjet3 << ", " << cut_pt_bjet << ", " << cut_pt_bjet << ", " << cut_pt_bjet <<  "]\t\t\t" << pass_ptb << endl;
  cout << "chisq minimum < " << cut_chisq_min << "\t\t\t\t\t\t" << pass_chisq << endl;
  cout << "DeltaM(min,med,max) < [" << cut_DeltaM_min << ", " << cut_DeltaM_med << ", " << cut_DeltaM_max << "]\t\t\t\t\t" << pass_DeltaM << endl;
  cout << "Three reco Higgses with pT > [" << cut_pt_higgs1 << ", " << cut_pt_higgs2 << ", " << cut_pt_higgs3 << "]\t\t\t" << pass_pthiggses << endl;
  cout << "DeltaR(b,b) in reco Higgses < " << cut_dR_hbbreco << "\t\t\t\t" << pass_dRbbhiggses << endl;
  cout << "dR between reco Higgses < " << cut_dR_higgses << "\t\t\t\t\t" << pass_dRhiggses << endl;
  cout << "6bs with dR(b,b) > " << cut_dRbbmin << "\t\t\t\t\t\t" << pass_drbb << endl;
  cout << "------------------" << endl;
  cout << "total weight out =\t\t\t\t\t\t" <<  passcuts << endl;
  cout << "actual MC events = \t\t\t\t\t\t" << eventcount << endl;
  cout << "efficiency =\t\t\t\t\t\t\t" <<  passcuts/total_weight_in << endl;
  cout << "------------------" << endl;
  outdat << passcuts/total_weight_in << endl;
  return 0;
}


//check if an integer is in the vector<int>:
bool is_in(int a, vector<int> intvec) {
  bool is_it_in = 0;
  for(int iv = 0; iv < intvec.size(); iv++) {
    if(intvec[iv] == a) {
      is_it_in = 1;
    }
  }
  return is_it_in;
}


double analyze_event(fastjet::PseudoJet photon1, fastjet::PseudoJet photon2, fastjet::PseudoJet cjet, fastjet::PseudoJet bjet, fastjet::PseudoJet lepton, fastjet::PseudoJet etmiss, double evweight_i) {
  

  //construct all relevant variables:
  

  //passed_mcevents++;


  return evweight_i;
}


double Pb_to_b(double pt) {
  if(perfect_tagging) { return 1.0; }
  return 1.00;//0.75;
}

double Pb_to_c(double pt) {
  if(perfect_tagging) { return 0.; } 
  return 0.125; 
}

double Pc_to_c(double pt) {
  if(perfect_tagging) { return 1.0; }
  return 0.2;
}

double Pc_to_b(double pt) {
  if(perfect_tagging) { return 0.0; }
  return 0.1;
}

double Pjet_to_b(double pt) {
  if(perfect_tagging) { return 0.0; }
  return 0.01;
}

double Pjet_to_c(double pt) {
  if(perfect_tagging) { return 0.0; }
  return 0.005;
}

// jet to lepton mistag prob
double Pjet_to_photon(double pt) {
  if(perfect_tagging) { return 0.0; }
  double pval(0.);
  double alpha = 0.01;
  double beta = 1/30.0;

  pval = alpha*exp(-beta * pt);

  return pval;
}

double ctag_weight(fastjet::PseudoJet jet, bool btag, bool ctag) {
  double weight = 0;
  if(btag) { 
    weight = Pb_to_c(jet.perp());
    return weight; 
  }
  else if(ctag) { 
    weight = Pc_to_c(jet.perp());
    return weight;
  }
  else { // light jet
    weight = Pjet_to_c(jet.perp());
    return weight; 
  }
  return weight; 
}


double btag_weight(fastjet::PseudoJet jet, bool btag, bool ctag) {
  double weight = 0;
  if(btag) { 
    weight = Pb_to_b(jet.perp());
    return weight; 
  }
  else if(ctag) { 
    weight = Pc_to_b(jet.perp());
    return weight;
  }
  else { // light jet
    weight = Pjet_to_b(jet.perp());
    return weight; 
  }
  return weight; 
}

double atag_weight(fastjet::PseudoJet jet, bool btag, bool ctag) {
  double weight = 0;
  if(btag) { 
    weight = 0.;
  }
  else if(ctag) { 
    weight = 0.;
  }
  else {
    weight = Pjet_to_photon(jet.perp());
  }
  return weight; 
}


double dot(fastjet::PseudoJet p1, fastjet::PseudoJet p2) {
  return (p1.e() * p2.e() - p1.px() * p2.px() - p1.py() * p2.py() - p1.pz() * p2.pz() );
}


double deltaR(fastjet::PseudoJet p1, fastjet::PseudoJet p2) { 
  double dphi_tmp; 

  dphi_tmp = p2.phi() - p1.phi();
  if(dphi_tmp > M_PI) 
    dphi_tmp = 2 * M_PI - dphi_tmp;
  else if( dphi_tmp < - M_PI)  
    dphi_tmp = 2 * M_PI + dphi_tmp;
  
  //  return sqrt(sqr(p1.eta() - p2.eta()) + sqr(dphi_tmp));
  return sqrt(sqr(p1.rap() - p2.rap()) + sqr(dphi_tmp));
}

//----------------------------------------------------------------------
// does the actual work for printing out a jet
//----------------------------------------------------------------------
ostream & operator<<(ostream & ostr, const PseudoJet & jet) {
  ostr << "e, pt, y, phi =" 
       << " " << setw(10) <<  jet.e()  
       << " " << setw(10) << jet.perp() 
       << " " << setw(6) <<  jet.rap()  
       << " " << setw(6) <<  jet.phi()  
       << ", mass = " << setw(10) << jet.m()
       << ", btag = " << jet.user_index();
  return ostr;
}
char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}


bool btag_hadrons(fastjet::PseudoJet jet) {
  bool btagged(false);
  /* search constintuents of jets for b-mesons */
  for(int cc = 0; cc < jet.constituents().size(); cc++) { 
    for(int bb = 0; bb < 105; bb++) { 
      if(jet.constituents()[cc].user_index() == bhadronid[bb]) { 
	btagged = true;
	//	cout << "Jet B-tagged!" << endl;
	//	cout << jet << endl;
      }
    }
  }
  return btagged;
}

std::pair<complex_ld, complex_ld> pnuz_fromw(fastjet::PseudoJet plepton, fastjet::PseudoJet pmiss) {
  
  double l = sqrt(sqr(plepton.px()) + sqr(plepton.py()) + sqr(plepton.pz()));
  double A = pow(mw,2)/2. + plepton.px()*pmiss.px() + plepton.py()*pmiss.py();
  double zl = plepton.pz();
  double a = pow(l,2) - pow(zl,2);
  double b = - 2 * A * zl;
  double c = - pow(A,2) + pow(l,2) * pow(pmiss.perp(),2);
  
  std::pair<complex_ld, complex_ld> result = quadsolve(a,b,c);
  return result;
}

std::pair<fastjet::PseudoJet, fastjet::PseudoJet> get_Wvectors(fastjet::PseudoJet plepton, fastjet::PseudoJet pmiss) { 
  std::pair<fastjet::PseudoJet, fastjet::PseudoJet> wvecs;

  //get the two solutions for the pnuz
  std::pair<complex_ld, complex_ld> pnuz_fromw_res = pnuz_fromw(plepton, pmiss);

  //reconstruct the two ws
  double Ew1 = sqrt( pow((plepton.px()+pmiss.px()),2) + pow(plepton.py()+pmiss.py(),2) + pow(plepton.pz() + real(pnuz_fromw_res.first),2) + pow(mw,2) );
  double Ew2 = sqrt( pow((plepton.px()+pmiss.px()),2) + pow(plepton.py()+pmiss.py(),2) + pow(plepton.pz() + real(pnuz_fromw_res.second),2) + pow(mw,2) );
    
  fastjet::PseudoJet wvec1 = fastjet::PseudoJet( plepton.px()+pmiss.px(), plepton.py()+pmiss.py(), plepton.pz() + real(pnuz_fromw_res.first), Ew1 );
  fastjet::PseudoJet wvec2 = fastjet::PseudoJet( plepton.px()+pmiss.px(), plepton.py()+pmiss.py(), plepton.pz() + real(pnuz_fromw_res.second), Ew2 );
  wvecs.first = wvec1;
  wvecs.second = wvec2;
  return wvecs; 
}


std::pair<fastjet::PseudoJet, fastjet::PseudoJet> get_Wvectors_GTX(fastjet::PseudoJet plepton, fastjet::PseudoJet pmiss) { 
  std::pair<fastjet::PseudoJet, fastjet::PseudoJet> wvecs;

  //get the two solutions for the pnuz
  std::pair<complex_ld, complex_ld> pnuz_fromw_res = pnuz_fromw(plepton, pmiss);

  fastjet::PseudoJet pnu1 = fastjet::PseudoJet( pmiss.px(), pmiss.py(), real(pnuz_fromw_res.first), sqrt( sqr(pmiss.px()) + sqr(pmiss.py()) + sqr(real(pnuz_fromw_res.first)) ) );
  fastjet::PseudoJet pnu2 = fastjet::PseudoJet( pmiss.px(), pmiss.py(), real(pnuz_fromw_res.second), sqrt( sqr(pmiss.px()) + sqr(pmiss.py()) + sqr(real(pnuz_fromw_res.second)) ) );

  fastjet::PseudoJet wvec1 = pnu1 + plepton;
  fastjet::PseudoJet wvec2 = pnu2 + plepton;
    
  wvecs.first = wvec1;
  wvecs.second = wvec2;
  return wvecs; 
}
	



fastjet::PseudoJet smear_jet(fastjet::PseudoJet jet_in) {
  if(donotsmear_jets) { return jet_in; }

  fastjet::PseudoJet smeared; 
  double smearing = 20, smeared_pt(0);

  double pt = jet_in.perp();
  double eta = fabs(jet_in.eta());
  double sigma(0);
  
  //double a, b, S, C;
  //if(eta < 0.8) { a = 3.2; b = 0.07; S = 0.74; C = 0.05; }
  //if(eta > 0.8 && eta < 1.2) { a = 3.0; b = 0.07; S = 0.81; C = 0.05; }
  //if(eta > 1.2 && eta < 2.8) { a = 3.3; b = 0.08; S = 0.54; C = 0.05; }
  //if(eta > 2.8 /*&& eta < 3.6*/) { a = 2.8; b = 0.11; S = 0.83; C = 0.05; }

  //double mu_pileup = 40;
  //double N = a + b * mu_pileup;

  //sigma = pt * sqrt( sqr(N)/sqr(pt) + sqr(S) / pt + sqr(C) );*/

  //MRM:
  sigma = 1.0 * sqrt(pt);
  
  smeared_pt = fabs(rnd.Gaus(0,sigma));
  double theta = rnd.Rndm()*M_PI;
  double phi = rnd.Rndm()*2.*M_PI;

  
  double deltaE = - jet_in.e() + sqrt( sqr(jet_in.e()) + sqr(smeared_pt) + 2 * (smeared_pt*sin(theta)*cos(phi)*jet_in.px() + smeared_pt*sin(theta)*sin(phi)*jet_in.py() + smeared_pt*cos(theta)*jet_in.pz()));

  fastjet::PseudoJet smearing_vector(smeared_pt*sin(theta)*cos(phi),smeared_pt*sin(theta)*sin(phi), smeared_pt*cos(theta), deltaE);
  
  smeared = jet_in + smearing_vector;  
  
  return smeared;
}

fastjet::PseudoJet smear_photon(fastjet::PseudoJet photon_in) {
  if(donotsmear_photons) { return photon_in; }

  fastjet::PseudoJet smeared;
  double smeared_pt = 0;
  //double smear_frac = 0.1E-2;
  //double smear_sampling = 0.15;
  //FROM MRM:
  double smear_frac = 0.17E-2;
  double smear_sampling = 0.20;
  double sigma(smear_sampling * sqrt(photon_in.perp()) + smear_frac*photon_in.perp());

  smeared_pt = fabs(rnd.Gaus(0,sigma));
  double theta = rnd.Rndm()*M_PI;
  double phi = rnd.Rndm()*2.*M_PI;

  //cout << smeared_pt*sin(theta)*cos(phi) <<  " " << smeared_pt*sin(theta)*sin(phi) << " " << smeared_pt*cos(theta) << " " << smeared_pt << endl;
  //cout << smeared_pt*sin(theta)*cos(phi) << " " << smeared_pt*sin(theta)*sin(phi) << "  " <<  smeared_pt*cos(theta) << endl;

  
  double deltaE = - photon_in.e() + sqrt( sqr(photon_in.e()) + sqr(smeared_pt) + 2 * (smeared_pt*sin(theta)*cos(phi)*photon_in.px() + smeared_pt*sin(theta)*sin(phi)*photon_in.py() + smeared_pt*cos(theta)*photon_in.pz()));
  
  fastjet::PseudoJet smearing_vector(smeared_pt*sin(theta)*cos(phi),smeared_pt*sin(theta)*sin(phi), smeared_pt*cos(theta), deltaE);
  smeared = photon_in + smearing_vector;
  //cout << "smeared mass = " << smeared.m() << endl;
  return smeared;

}



fastjet::PseudoJet smear_lepton(fastjet::PseudoJet lepton_in, int lepton_id) {

  if(donotsmear_leptons) { return lepton_in; }
   
    
  fastjet::PseudoJet smeared;
  double smeared_pt = 0;
  double smearing = 20.;

  double pt = lepton_in.perp();
  double lepton_energy = lepton_in.e();
  double eta = fabs(lepton_in.eta());
  double sigma(0);

  //see ATL-PHYS-PUB-2013-009
  if(lepton_id == 13) {
    double sigma_id = 0;
    double sigma_ms = 0;
    double sigma_cb = 0;
    double a1, a2, b0, b1, b2;
    
    if(eta < 0.18) { a1 = 0.01061; a2 = 0.000157; }
    if(eta > 0.18 && eta < 0.36) { a1 = 0.01084; a2 = 0.000153; }
    if(eta > 0.36 && eta < 0.54) { a1 = 0.01124; a2 = 0.000150; }
    if(eta > 0.54 && eta < 0.72) { a1 = 0.01173; a2 = 0.000149; }
    if(eta > 0.72 && eta < 0.90) { a1 = 0.01269; a2 = 0.000148; }
    if(eta > 0.90 && eta < 1.08) { a1 = 0.01406; a2 = 0.000161; }
    if(eta > 1.08 && eta < 1.26) { a1 = 0.01623; a2 = 0.000192; }
    if(eta > 1.26 && eta < 1.44) { a1 = 0.01755; a2 = 0.000199; } 
    if(eta > 1.44 && eta < 1.62) { a1 = 0.01997; a2 = 0.000232; } 
    if(eta > 1.62 && eta < 1.80) { a1 = 0.02453; a2 = 0.000261; }
    if(eta > 1.80 && eta < 1.98) { a1 = 0.03121; a2 = 0.000297; }
    if(eta > 1.98 && eta < 2.16) { a1 = 0.03858; a2 = 0.000375; }
    if(eta > 2.16 && eta < 2.34) { a1 = 0.05273; a2 = 0.000465; }
    if(eta > 2.34 && eta < 2.52) { a1 = 0.05329; a2 = 0.000642; }
    if(eta > 2.52 /*&& eta < 2.70*/) { a1 = 0.05683; a2 = 0.000746; }

    if(eta < 1.05) { b1 = 0.02676; b2 = 0.00012; }
    if(eta > 1.05) { b1 = 0.03880; b2 = 0.00016; }

    sigma_id = pt * sqrt( a1 + sqr(a2 * pt) );
    sigma_ms = pt * sqrt( sqr(b0/pt) + sqr(b1) + sqr(b2*pt) );
    sigma = (sigma_id * sigma_ms)/sqrt( sqr(sigma_id) + sqr(sigma_ms) ); //sigma_cb

  }


  if(lepton_id == 11) {
    double sigma = 0;
    if(eta < 1.4) { sigma = sqrt( sqr(0.3) + sqr(0.10 * sqrt(lepton_energy)) + sqr( 0.010 * lepton_energy ) ); }
    if(eta > 1.4 /* && eta < 2.47 */) { sigma = sqrt( sqr(0.3) + sqr(0.15 * sqrt(lepton_energy)) + sqr( 0.015 * lepton_energy ) ); }
  }

  smeared_pt = fabs(rnd.Gaus(0,sigma));
  double theta = rnd.Rndm()*M_PI;
  double phi = rnd.Rndm()*2.*M_PI;

  double deltaE = - lepton_in.e() +  sqrt( sqr(lepton_in.e()) + sqr(smeared_pt) + 2 * (smeared_pt*sin(theta)*cos(phi)*lepton_in.px() + smeared_pt*sin(theta)*sin(phi)*lepton_in.py() + smeared_pt*cos(theta)*lepton_in.pz()));

  fastjet::PseudoJet smearing_vector(smeared_pt*sin(theta)*cos(phi),smeared_pt*sin(theta)*sin(phi), smeared_pt*cos(theta), deltaE);
  
  smeared = lepton_in + smearing_vector;  
  
  //smeared.reset(smeared.px(), smeared.py(), smeared.pz(), eprime);

  smeared.set_user_index(lepton_id);
  
  return smeared;
}

bool lepton_efficiency_accept(fastjet::PseudoJet lepton_in, int lepton_id) {
  bool accepted(1);
  if(donot_apply_efficiency) { return accepted; }

  double pt = lepton_in.perp();
  double eta = fabs(lepton_in.eta());
  
  double epsilon = 0;
  if(lepton_id == 11) {
    epsilon = 0.85 - 0.191 * exp(1 - pt/20);
  }
  if(lepton_id == 13) {
    if(eta<0.1) { epsilon = 0.54; }
    if(eta>0.1) { epsilon = 0.97; } 
  }
  double random_num = rnd.Rndm();
  //  cout << lepton_id << " " << pt << " " << eta << " " << random_num << " " << epsilon << endl;
  if(random_num > epsilon) { accepted = 0; }
  return accepted;
}
bool photon_efficiency_accept(fastjet::PseudoJet photon_in) {
  bool accepted(1);
  if(donot_apply_efficiency) { return accepted; }

  double pt = photon_in.perp();
  double eta = fabs(photon_in.eta());
  
  double epsilon = 0;
  
  epsilon = 0.76 - 1.98 * exp(-pt/16.1);
 
  double random_num = rnd.Rndm();
  if(random_num > epsilon) { accepted = 0; }
  return accepted;
}
bool jet_efficiency_accept(fastjet::PseudoJet jet_in) {
    bool accepted(1);
    if(donot_apply_efficiency) { return accepted; }

    double pt = jet_in.perp();
    double epsilon = 0;

    epsilon = 0.75 + (0.95 - 0.75) * pt / (50. - 20.);
    if(epsilon < 0) { epsilon = 0; }
    if(epsilon > 1.0) { epsilon = 1.0; }

    
    double random_num = rnd.Rndm();
    if(random_num > epsilon) { accepted = 0; }
    return accepted;
    
}

                     
                     
std::pair<complex_d, complex_d> quadsolve(complex_ld a, complex_ld b, complex_ld c) {
  std::pair<complex_d, complex_d> res;
  complex_ld D(0.);
  complex_ld ac = a*c;
  complex_ld fourac = complex_ld(4 * real(ac), 4 * imag(ac));
  D =  sqrt(b*b - fourac);
  complex_ld twoa = complex_ld( 2 * real(a), 2 * imag(a));
  res.first = (-b + D)/twoa;
  res.second = (-b - D)/twoa;

  return res;
}

// start is the current position in the list, advancing by 2 each time
// pass 0 as start when calling at the top level
void generatePairings(int* items, int itemcount, int start)
{
  vector<int> items_complete;
    if(itemcount & 1)
        return; // must be an even number of items
    // is this a complete pairing?
    if(start == itemcount)
    {
        // output pairings:
        int i;
	//items_complete.resize(itemcount);
        for(i = 0; i<itemcount; i+=2)
        {
	  printf("[%d, %d] ", items[i], items[i+1]);
	  items_complete.push_back(items[i]);
	  items_complete.push_back(items[i+1]);
        }
	//global!
	pairs_of_six.push_back(items_complete);
        //printf("\n");
        return;
    }

    // for the next pair, choose the first element in the list for the
    // first item in the pair (meaning we don't have to do anything 
    // but leave it in place), and each of the remaining elements for
    // the second item:
    int j;
    for(j = start+1; j<itemcount; j++)
    {
        // swap start+1 and j:
        int temp = items[start+1];
        items[start+1] = items[j];
        items[j] = temp;

        // recurse:
        generatePairings(items, itemcount, start+2);

        // swap them back:
        temp = items[start+1];
        items[start+1] = items[j];
        items[j] = temp;
    }

}
