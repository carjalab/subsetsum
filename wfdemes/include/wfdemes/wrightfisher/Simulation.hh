/*  Simulation runner for the Wright-Fisher model on demes.
 *  
 *  :author: Dennis Aldea <dennisaldea@cmu.edu>
 */


#pragma once


#include <filesystem>
#include <optional>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/random.hpp>

#include <magic_enum/magic_enum.hpp>

#include <basemodel/Simulation.hh>
#include <wrightfisher/Parameters.hh>


namespace wrightfisher {

/*  A simulation of the Wright-Fisher model on demes.
 *  
 *  Attributes
 *  ----------
 *  parameters : const Parameters& (read-only)
 *      Parameter set passed to this simulation.
 *  seed : uint32_t (optional, read-only)
 *      Seed for random number generation; enables reproducible results.
 *  outfile : std::filesystem::path (optional, read-only)
 *      Output file path for simulation trajectory.
 *  status : sims::SimulationStatus (read-only)
 *      Simulation execution status.
 *  generation : int (read-only)
 *      Current simulation time (in generations).
 *  mutant_cnts : vector<int> (read-only)
 *      Current deme mutant counts.
 *  mutant_traj : vector<vector<int>> (read-only)
 *      Deme mutant counts trajectory.
 *  outcome : sims::SimulationOutcome (optional, read-only)
 *      Mutation fixation outcome. Only defined after simulation has ended.
 *  
 *  Methods
 *  -------
 *  mutant_total() const : int
 *      Total mutant count.
 */
class Simulation : public basemodel::Simulation {
  private:

    /*  Runtime Invariable Attributes */

    const Parameters& params_;
    std::optional<uint32_t> seed_;
    boost::random::xoshiro128pp rng;
    std::optional<std::filesystem::path> outfile_;

    /*  Runtime Variable Attributes */

    basemodel::SimulationStatus status;
    int generation_;
    std::vector<int> mutant_cnts_;
    std::vector<std::vector<int>> mutant_traj_;
    std::optional<basemodel::SimulationOutcome> outcome;

    /*  Make an RNG using either random 128-bit seed or given 32-bit seed. */
    void make_rng_from_seedvalue(uint32_t seedvalue);
    
    /*  Record current generation to simulation history. */
    void record_generation();

    /*  Print simulation trajectory to standard output. */
    void print_trajectory() const;
    
    /*  Write simulation trajectory to file. */
    void write_trajectory(std::filesystem::path trajectfile) const;

    /* Calculate migration flow matrix from deme populations. */
    boost::numeric::ublas::matrix<double> calc_migration_mtrx() const;

    boost::numeric::ublas::vector<double, std::vector<double>>
        calc_fitness_vctr() const;

    struct Reporter {
        /*  Format strings used by `str()`. */
        static inline const struct FmtStrs {
            std::string params = "Parameters\n{}";
            std::string seed = "RNG seed: {}";
            std::string outfile = "Trajectory file: {}";
            std::string status = "Run status: {}";
            std::string generation = "Generation: {}";
            std::string mutant_cnts = "Deme mutant counts: {}";
            std::string outcome = "Simulation outcome: {}";

            FmtStrs() {}
        } fmtstrs;
        static inline const std::string NoValue_repr = "(none)";

        std::string
            params, seed, outfile, status, generation, mutant_cnts, outcome;
        
        Reporter(const Simulation& simltn);
    };

  public:

    const decltype(params_)& params;
    const decltype(seed_)& seed;
    const decltype(outfile_)& outfile;
    const decltype(generation_)& generation;
    const decltype(mutant_cnts_)& mutant_cnts;
    const decltype(mutant_traj_)& mutant_traj;

    /*  Simulation constructor.
     *  
     *  Parameters
     *  ----------
     *  parameterset : const Parameters&
     *      Simulation parameters. Assigned to `params`.
     *  seedvalue : uint32_t (optional)
     *      Seed value; enables reproducible results. If `seedvalue` = 0, a
     *      random 128-bit seed value is used instead. Assigned to `seed` if 
     *      `seedvalue` != 0. (Default: 0)
     *  trajectfile : std::filesystem::path (optional)
     *      Output file path for simulation trajectory. Assigned to `outfile`.
     *      (Default: no output)
     */
    Simulation(const Parameters& parameterset);
    
    /*  Simulation constructor.
     *  
     *  Parameters
     *  ----------
     *  parameterset : const Parameters&
     *      Simulation parameters. Assigned to `params`.
     *  seedvalue : uint32_t (optional)
     *      Seed value; enables reproducible results. If `seedvalue` = 0, a
     *      random 128-bit seed value is used instead. Assigned to `seed` if 
     *      `seedvalue` != 0. (Default: 0)
     *  trajectfile : std::filesystem::path (optional)
     *      Output file path for simulation trajectory. Assigned to `outfile`.
     *      (Default: no output)
     */
    Simulation(const Parameters& parameterset, uint32_t seedvalue);
    
    /*  Simulation constructor.
     *  
     *  Parameters
     *  ----------
     *  parameterset : const Parameters&
     *      Simulation parameters. Assigned to `params`.
     *  seedvalue : uint32_t (optional)
     *      Seed value; enables reproducible results. If `seedvalue` = 0, a
     *      random 128-bit seed value is used instead. Assigned to `seed` if 
     *      `seedvalue` != 0. (Default: 0)
     *  trajectfile : std::filesystem::path (optional)
     *      Output file path for simulation trajectory. Assigned to `outfile`.
     *      (Default: no output)
     */
    Simulation(
        const Parameters& parameterset, uint32_t seedvalue,
        std::filesystem::path trajectfile);
    
    /*  The parameter set passed to this simulation.
     *  
     *  Returns
     *  -------
     *  const WFParameterSet&
     */
    const Parameters& get_parameters() const override;

    /*  The execution status of this simulation run.
     * 
     *  Returns
     *  -------
     *  SimulationStatus
     *      [TODO: document return values]
     */
    basemodel::SimulationStatus get_status() const override;
    
    /*  The current time (i.e., generation) in this simulation run.
     *  
     *  Returns
     *  -------
     *  int
     */
    int get_time() const override;

    /*  The current state (i.e., deme mutant counts) of this simulation run.
     *  
     *  Returns
     *  -------
     *  const std::vector<int>&
     */
    const std::vector<int>& get_state() const override;
    
    /*  The history (i.e., state trajectory) of this simulation run.
     *  
     *  Returns
     *  -------
     *  std::vector<std::vector<int>>
     */
    const std::vector<std::vector<int>>& get_history() const override;

    /*  The outcome (i.e, mutation fate) of this simulation run.
     *  
     *  Note that the outcome can only be queried after the simulation has
     *  ended.
     *  
     *  Returns
     *  -------
     *  sims::SimulationOutcome
     *      [TODO: document return values]
     *  
     *  Raises
     *  ------
     *  std::domain_error
     *      If called before the simulation has ended.
     */
    basemodel::SimulationOutcome get_outcome() const override;

    /*  Calculate total mutant count.
     *  
     *  Returns
     *  -------
     *  int
     *      Total mutant count.
     */
    int mutant_total() const;

    /*  Run simulation for up to `maxgens` generations.
     *  
     *  Parameters
     *  ----------
     *  maxgens : int
     *      Maximum number of generations to be simulated.
     */
    void run(int maxgens) override;

    /*  Report all attributes of this simulation run in human-readable form.
     *  
     *  Returns
     *  -------
     *  std::string
     *      A formatted report of the attributes of this simulation run.
     */
    std::string str() const override;
};

}
