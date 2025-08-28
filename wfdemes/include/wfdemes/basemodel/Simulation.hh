/*  Abstract interface for simulation runner classes.
 *  
 *  :author: Dennis Aldea <dennisaldea@cmu.edu>
 */


#pragma once


#include <stdexcept>
#include <string>
#include <vector>

#include <basemodel/Parameters.hh>


namespace basemodel {

enum class SimulationStatus { Pending, Running, Ended };

enum class SimulationOutcome { Extinct, Coexistent, Fixed };

struct SimulationResult {
    SimulationOutcome outcome;
    int duration;
};

/*  Virtual base class for simulation runner classes.
 *  
 *  Methods
 *  -------
 *  get_parameters() : const Parameters& (virtual, const)
 *      Parameter set passed to this simulation.
 *  get_status() : SimulationStatus (virtual, const)
 *      Execution status of this simulation.
 *  get_time() : int (virtual, const)
 *      Current time in this simulation.
 *  get_state() : const std::vector<int>& (virtual, const)
 *      Current state (i.e., mutant load) of this simulation.
 *  get_history() : const std::vector<std::vector<int>>& (virtual, const)
 *      History (i.e., state trajectory) of this simulation.
 *  get_outcome() : SimulationOutcome (virtual, const)
 *      Outcome (i.e., mutation fate) of this simulation.
 *  get_result() : SimulationResult (const)
 *      Result (i.e., outcome and duration) for this simulation.
 *  run(int maxgens) : void (virtual)
 *      Run this simulation for up to `maxgen` generations.
 *  str() : std::string (virtual, const)
 *      Report all attributes of this simulation.
 */
class Simulation {
  public:
    
    /*  The parameter set passed to this simulation.
     *  
     *  Returns
     *  -------
     *  const Parameters&
     */
    virtual const Parameters& get_parameters() const = 0;

    /*  The current execution status of this simulation run.
     *  
     *  The execution status of a simulation is one of the following:
     *    - `Pending`: If the simulation has been initialized but not yet run.
     *    - `Running`: If the simulation run has begun but not yet ended.
     *    - `Ended`: If the simulation run has ended.
     * 
     *  Returns
     *  -------
     *  SimulationStatus
     */
    virtual SimulationStatus get_status() const = 0;

    /*  The current time in this simulation run.
     *  
     *  Returns
     *  -------
     *  int
     */
    virtual int get_time() const = 0;

    /*  The current state (i.e., mutant load) of this simulation run.
     *  
     *  Returns
     *  -------
     *  const std::vector<int>&
     */
    virtual const std::vector<int>& get_state() const = 0;
    
    /*  The history (i.e., state trajectory) of this simulation run.
     *  
     *  Returns
     *  -------
     *  const std::vector<std::vector<int>>&
     */
    virtual const std::vector<std::vector<int>>& get_history() const = 0;

    /*  The outcome (i.e, mutation fate) of this simulation run.
     *  
     *  The outcome of a simulation run is one of the following:
     *    - `Extinct`: If the mutation went to extinction.
     *    - `Coexistent`: If the mutation neither went to extinction nor
     *      reached fixation before the end of the simulation run.
     *    - `Fixed`: If the mutation reached fixation.
     *    
     *  Note that a simulation's outcome can only be queried after the
     *  simulation run has ended.
     *  
     *  Returns
     *  -------
     *  SimulationOutcome
     *  
     *  Raises
     *  ------
     *  std::domain_error
     *      If called before the simulation run has ended.
     */
    virtual SimulationOutcome get_outcome() const = 0;
    
    /*  The result (i.e. outcome and duration) of this simulation run.
     *  
     *  The result of a simulation run consists of:
     *    - the run's mutation fate
     *    - the run's duration
     *  
     *  Note that a simulation's result can only be queried after the
     *  simulation run has ended.
     *  
     *  Returns
     *  -------
     *  SimulationResult
     * 
     *  Raises
     *  ------
     *  std::domain_error
     *      If called before the simulation has ended.
     */
    SimulationResult get_result() const {
        const std::string NOTENDED_ERRMSG = "Simulation has not yet ended.";
        // check that simulation ended
        bool simltn_ended = (get_status() == SimulationStatus::Ended);
        if (!simltn_ended) { throw std::domain_error(NOTENDED_ERRMSG); }
        // duration is the time at simulation end
        return SimulationResult{get_outcome(), get_time()};
    }

    /*  Run this simulation for up to `maxgens` generations.
     *  
     *  Parameters
     *  ----------
     *  maxgens : int
     *      Maximum number of generations to be simulated.
     */
    virtual void run(int maxgens) = 0;

    /*  Report all attributes of this simulation run in human-readable form.
     *  
     *  Returns
     *  -------
     *  std::string
     *      A formatted report of the attributes of this simulation run.
     */
    virtual std::string str() const = 0;
};

}
