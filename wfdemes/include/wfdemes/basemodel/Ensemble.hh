/*  Abstract interface for ensemble launcher classes.
 *  
 *  :author: Dennis Aldea <dennisaldea@cmu.edu>
 */


#pragma once


#include <string>
#include <vector>

#include <fmt/format.h>

#include <basemodel/Parameters.hh>
#include <basemodel/Simulation.hh>


namespace basemodel {

/*  Data structure containing simulation counts for an ensemble launcher.*/
struct EnsembleStatus {
    /*  Format string used by `str()`. */
    inline static const std::string FMTSTR =
        "{} pending, {} running, {} ended";

    /*  Number of simulations that have not yet started. */
    int pending;
    /*  Number of simulations that have started but have not yet ended. */
    int running;
    /*  Number of simulations that have ended. */
    int ended;

    /*  Report an EnsembleStatus in human-readable form.
     *  
     *  Returns
     *  -------
     *  std::string
     *      A formatted report of the EnsembleStatus.
     */
    std::string str() const {
        std::string report = fmt::format(FMTSTR, pending, running, ended);
        return report;
    }
};

/*  Virtual base class for ensemble launcher classes.
 *  
 *  Methods
 *  -------
 *  get_parameters() : const Parameters& (virtual, const)
 *      Parameter set passed to simulations in this ensemble.
 *  get_status() : EnsembleStatus (virtual, const)
 *      Execution status of this ensemble.
 *  get_results() : const std::vector<SimulationResult>& (virtual, const)
 *      Results for all simulation runs in this ensemble.
 *  run(int numruns, int maxgens) : void (virtual)
 *      Run ensemble of `numruns` simulations for up to `maxgen` generations.
 *  str() : std::string (virtual, const)
 *      Report all attributes of this ensemble launcher.
 */
class Ensemble {
  public:

    /*  The parameter set passed to simulations in this ensemble.
     *  
     *  Returns
     *  -------
     *  const Parameters&
     */
    virtual const Parameters& get_parameters() const = 0;

    /*  The current execution status of this ensemble.
     *  
     *  The execution status of an ensemble launcher consists of:
     *    - the number of simulations pending
     *    - the number of simulations running
     *    - the number of simulations ended
     *  
     *  Returns
     *  -------
     *  EnsembleStatus
     */
    virtual EnsembleStatus get_status() const = 0;

    /*  The results for all simulation runs in this ensemble.
     *  
     *  Note that the results of an ensemble can only be queried after all
     *  simulations runs in the ensemble have ended.
     *  
     *  Returns
     *  -------
     *  const std::vector<SimulationResult>&
     *  
     *  Raises
     *  ------
     *  std::domain_error
     *      If called before all simulation runs are ended.
     *  
     *  See Also
     *  --------
     *  `Simulation::get_result()`
     */
    virtual const std::vector<SimulationResult>& get_results() const = 0;

    /*  Run `numsims` simulations for up to `maxgen` generations.
     *  
     *  Parameters
     *  ----------
     *  numsims : int
     *      Number of simulations to run.
     *  maxgens: int
     *      Maximum number of generations to simulate in each run.
     */
    virtual void run(int numsims, int maxgens) = 0;

    /*  Report all attributes of this ensemble launcher in human-readable form.
     *  
     *  Returns
     *  -------
     *  std::string
     *      A formatted report of the attributes of this ensemble launcher.
     */
    virtual std::string str() const = 0;
};

}
