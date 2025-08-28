/*  Parameter set for the Wright-Fisher model on demes.
 *  
 *  :author: Dennis Aldea <dennisaldea@cmu.edu>
 */


#pragma once


#include <vector>

#include <fmt/format.h>

#include <basemodel/Parameters.hh>


namespace wrightfisher {

/*  Calculate deme populations from total population and deme weights.
 *  
 *  Parameters
 *  ----------
 *  totalpop : int
 *      Total population.
 *  demeweights : std::vector<double>
 *      Deme weights.
 *  
 *  Returns
 *  -------
 *  std::vector<int>
 *      Deme populations.
*/
std::vector<int> calc_deme_pops(int totalpop, std::vector<double> demeweights);

/*  Calculate mutant counts from deme populations and mutant frequency.
 *  
 *  Parameters
 *  ----------
 *  demepops : std::vector<int>
 *      Deme populations.
 *  mutfreq : double
 *      Mutant frequency.
 *  
 *  Returns
 *  -------
 *  std::vector<int>
 *      Deme mutant counts.
 */
std::vector<int> calc_mutant_cnts(std::vector<int> demepops, double mutfreq);

/*  Calculate relative mutant fitness based on frequency-dependent selection.
 *  
 *  Parameters
 *  ----------
 *  ecolsel : double
 *      Ecological selection coefficient (α).
 *  mutfreq : double
 *      Mutant frequency.
 *  
 *  Returns
 *  -------
 *  double
 *      Mutant relative fitness.
 */
double calc_mutant_relfit(double ecolsel, double mutfreq);

/*  Parameter set for the Wright-Fisher model on demes.
 *  
 *  Attributes
 *  ----------
 *  total_pop : int (read-only)
 *      Total population size.
 *  num_demes : int (read-only)
 *      Number of demes.
 *  deme_pops : std::vector<int> (read-only)
 *      Population size of each deme.
 *  init_muts : std::vector<int> (read-only)
 *      Initial mutant count in each deme.
 *  ecol_selec : double (read-only)
 *      Ecological selection coefficient (α).
 *  intr_selec : double (read-only)
 *      Intrinsic selection coefficient (s).
 *  migrt_rate : int (read-only)
 *      Migration rate between demes (m).
 *  
 *  Methods
 *  -------
 *  str() : std::string
 *      Report all attributes in this parameter set.
 */
class Parameters : public basemodel::Parameters {
  private:
    int total_pop_;
    int num_demes_;
    std::vector<int> deme_pops_;
    std::vector<int> init_muts_;
    double ecol_selec_;
    double intr_selec_;
    double migrt_rate_;
    
    struct Reporter {
        /*  Format strings used by `str()`. */
        static inline const struct FmtStrs {
            std::string total_pop = "Total population: {}";
            std::string num_demes = "Number of demes: {}";
            std::string deme_pops = "Deme populations: {}";
            std::string init_muts = "Initial mutant counts: {}";
            std::string ecol_selec = "Ecological selection coeff.: {}";
            std::string intr_selec = "Intrinsic selection coeff.: {}";
            std::string migrt_rate = "Migration rate: {}";
            
            FmtStrs() {}
        } fmtstrs;
        
        std::string
            total_pop, num_demes, deme_pops, init_muts, ecol_selec, intr_selec,
            migrt_rate;

        Reporter(const Parameters& params);
    };

  public:
    const decltype(total_pop_)& total_pop;
    const decltype(num_demes_)& num_demes;
    const decltype(deme_pops_)& deme_pops;
    const decltype(init_muts_)& init_muts;
    const decltype(ecol_selec_)& ecol_selec;
    const decltype(intr_selec_)& intr_selec;
    const decltype(migrt_rate_)& migrt_rate;

    /*  Construct the parameter set for the Wright-Fisher model on demes.
     *  
     *  Parameters
     *  ----------
     *  totalpop : int
     *      Total population. Assigned to `total_pop`.
     *  demeweights : std::vector<double>
     *      Deme weights. Normalized to determine deme ratios.
     *  initmutfreq : double
     *      Initial mutant frequency. Determines initial mutant counts.
     *  ecolselec : double
     *      Ecological selection coefficient. Assigned to `ecol_selec`.
     *  intrselec : double
     *      Intrinsic selection coefficient. Assigned to `intr_selec`.
     *  migrtrate : double
     *      Migration rate. Assigned to `migration`.
     */
    Parameters(
        int totalpop, std::vector<double> demeweights, double initmutfreq,
        double ecolselec, double intrselec, double migrtrate);

    /*  Report all attributes in this parameter set in human-readable form.
     *  
     *  Returns
     *  -------
     *  std::string
     *      A formatted report of the attributes in this parameter set.
     */
    std::string str() const override;
};

}
