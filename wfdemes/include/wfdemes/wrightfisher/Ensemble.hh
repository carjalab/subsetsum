/*  Ensemble launcher for the Wright-Fisher model on demes.
 *  
 *  :author: Dennis Aldea <dennisaldea@cmu.edu>
 */


#pragma once


#include <cstdint>
#include <filesystem>
#include <optional>

#include <boost/random.hpp>

#include <fmt/format.h>

#include <basemodel/Ensemble.hh>
#include <basemodel/Simulation.hh>
#include <wrightfisher/Parameters.hh>


namespace wrightfisher {

/*  Ensemble launcher for a set of Wright-Fisher simulations.
 *  
 *  Attributes
 *  ----------
 *  params : Parameters (read-only)
 *      Parameter set shared by all simulations in this ensemble.
 *  seed : uint32_t (optional, read-only)
 *      Seed for random number generation; enables reproducible results.
 *  outfile : std::filesystem::path (optional, read-only)
 *      Output file for ensemble results.
 *  ntrajs : int (read-only)
 *      Number of simulation runs to be output to trajectory files.
 *  outdir : std::string (optional, read-only)
 *      Output directory for trajectory files. Required if `ntrajs` > 0.
 *  status : sims::EnsembleStatus (read-only)
 *      Execution status of this ensemble: simulations pending, started, ended.
 */
class Ensemble : public basemodel::Ensemble {
  private:

    /*  Runtime Invariable Attributes */
    
    Parameters params_;
    std::optional<uint32_t> seed_;
    boost::random::xoshiro128pp seeder_rng;
    std::optional<std::filesystem::path> outfile_;
    int ntrajs_;
    std::optional<std::filesystem::path> outdir_;

    /*  Runtime Variable Attributes */

    basemodel::EnsembleStatus status;
    std::vector<basemodel::SimulationResult> results;
    
    /*  Make trajectory output directory, raise error if it already exists. */
    static void create_outdir(std::filesystem::path trajectsdir);

    /*  Make an RNG using either random 128-bit seed or given 32-bit seed. */
    void make_rng_from_seedvalue(uint32_t seedvalue);
    
    /*  Print ensemble results to standard output. */
    void print_results() const;
    
    /*  Write ensemble results to file. */
    void write_results(std::filesystem::path resultsfile) const;

    struct Reporter {
        /*  Format strings used by `str()`. */
        static inline const struct FmtStrs{
            std::string params = "Parameters\n{}";
            std::string seed = "RNG seed: {}";
            std::string outfile = "Results file: {}";
            std::string ntrajs = "Number of trajectories: {}";
            std::string outdir = "Trajectories directory: {}";
            std::string status = "Ensemble status: {}";

            FmtStrs() {}
        } fmtstrs;
        static inline const std::string NoValue_repr = "(none)";

        std::string params, seed, outfile, ntrajs, outdir, status;

        Reporter(const Ensemble& ensmbl);
    };

  public:

    /*  Class Constants */

    /*  Trajectory filename format string. */
    inline static const std::string TRAJECTORY_FNAME_FMT = "traj_{}.csv";
    /*  Default filename for ensemble results output. */
    inline static const std::string RESULTS_FNAME_DEFAULT = "outcomes.csv";
    
    const decltype(params_)& params;
    const decltype(seed_)& seed;
    const decltype(outfile_)& outfile;
    const decltype(ntrajs_)& ntrajs;
    const decltype(outdir_)& outdir;

    /*  Ensemble constructor.
     *  
     *  Parameters
     *  ----------
     *  parameterset : Parameters
     *      Simulation parameters. Assigned to `params`.
     *  seedvalue : uint32_t (optional)
     *      Seed value; enables reproducible results. If `seedvalue` = 0, a
     *      random 128-bit seed value is used instead. Assigned to `seed` if 
     *      `seedvalue` != 0. (Default: 0)
     *  resultsfile : std::filesystem::path (optional)
     *      Output file for ensemble results. Assigned to `outfile`.
     *      (Default: none – write to standard output)
     *  numtrajects : int (optional)
     *      Number of trajectories to be output. Assigned to `ntrajs`.
     *      (Default: 0)
     *  trajectsdir : std::filesystem::path (optional)
     *      Output directory path. Assigned to `outdir`. (Note: required if
     *      `numtrajects` is specified.)
     */
    Ensemble(Parameters parameterset);
    
    /*  Ensemble constructor.
     *  
     *  Parameters
     *  ----------
     *  parameterset : Parameters
     *      Simulation parameters. Assigned to `params`.
     *  seedvalue : uint32_t (optional)
     *      Seed value; enables reproducible results. If `seedvalue` = 0, a
     *      random 128-bit seed value is used instead. Assigned to `seed` if 
     *      `seedvalue` != 0. (Default: 0)`
     *  resultsfile : std::filesystem::path (optional)
     *      Output file for ensemble results. Assigned to `outfile`.
     *      (Default: none – write to standard output)
     *  numtrajects : int (optional)
     *      Number of trajectories to be output. Assigned to `ntrajs`.
     *      (Default: 0)
     *  trajectsdir : std::filesystem::path (optional)
     *      Output directory path. Assigned to `outdir`. (Note: required if
     *      `numtrajects` is specified.)
     */
    Ensemble(Parameters parameterset, uint32_t seedvalue);

    /*  Ensemble constructor.
     *  
     *  Parameters
     *  ----------
     *  parameterset : Parameters
     *      Simulation parameters. Assigned to `params`.
     *  seedvalue : uint32_t (optional)
     *      Seed value; enables reproducible results. If `seedvalue` = 0, a
     *      random 128-bit seed value is used instead. Assigned to `seed` if 
     *      `seedvalue` != 0. (Default: 0)`
     *  resultsfile : std::filesystem::path (optional)
     *      Output file for ensemble results. Assigned to `outfile`.
     *      (Default: none – write to standard output)
     *  numtrajects : int (optional)
     *      Number of trajectories to be output. Assigned to `ntrajs`.
     *      (Default: 0)
     *  trajectsdir : std::filesystem::path (optional)
     *      Output directory path. Assigned to `outdir`. (Note: required if
     *      `numtrajects` is specified.)
     */
    Ensemble(
        Parameters parameterset, uint32_t seedvalue,
        std::filesystem::path resultsfile);
    
    /*  Ensemble constructor.
     *  
     *  Parameters
     *  ----------
     *  parameterset : Parameters
     *      Simulation parameters. Assigned to `params`.
     *  seedvalue : uint32_t (optional)
     *      Seed value; enables reproducible results. If `seedvalue` = 0, a
     *      random 128-bit seed value is used instead. Assigned to `seed` if 
     *      `seedvalue` != 0. (Default: 0)`
     *  resultsfile : std::filesystem::path (optional)
     *      Output file for ensemble results. Assigned to `outfile`.
     *      (Default: none – write to standard output)
     *  numtrajects : int (optional)
     *      Number of trajectories to be output. Assigned to `ntrajs`.
     *      (Default: 0)
     *  trajectsdir : std::filesystem::path (optional)
     *      Output directory path. Assigned to `outdir`. (Note: required if
     *      `numtrajects` specified.)
     */
    Ensemble(
        Parameters parameterset, uint32_t seedvalue,
        std::filesystem::path resultsfile, int numtrajects,
        std::filesystem::path trajectsdir);

    /*  The parameter set passed to simulations in this ensemble.
     *  
     *  Returns
     *  -------
     *  const Parameters&
     */
    const Parameters& get_parameters() const override;

    /*  The current execution status of this ensemble.
     *  
     *  An ensemble's execution status consists of the number of simulations
     *  pending, the number of simulations currently running, and the number of
     *  simulations ended.
     *  
     *  Returns
     *  -------
     *  EnsembleStatus
     */
    basemodel::EnsembleStatus get_status() const override;

    /*  The results (outcome & duration) for all simulations in this ensemble.
     *  
     *  Note that results can not be queried before all simulations have ended.
     *  
     *  Returns
     *  -------
     *  const std::vector<SimulationResult>&
     *  
     *  Raises
     *  ------
     *  std::domain_error
     *      If called before all simulations are ended.
     *  
     *  See Also
     *  --------
     *  `Simulation::get_result()`
     */
    const std::vector<basemodel::SimulationResult>& get_results()
        const override;

    /*  Run an ensemble of `nruns` simulations for up to `maxgen` generations.
     *  
     *  Simulation output is written to files created in the `outdir`
     *  directory; `outdir` is created if it does not exist.
     *  
     *  Parameters
     *  ----------
     *  numruns : int
     *      Number of simulations to run in the ensemble.
     *  maxgens: int
     *      Maxiumum number of generations to simulate in each run.
     *  
     *  Raises
     *  ------
     *  std::invalid_argument
     *      If `outdir` already exists.
     *  std::filesystem::filesystem_error
     *      If `outdir` does not exist but cannot be created.
     */
    void run(int numruns, int maxgens) override;

    /*  Report all attributes of this ensemble launcher in human-readable form.
     *  
     *  Returns
     *  -------
     *  std::string
     *      A formatted report of the attributes of this ensemble launcher.
     */
    std::string str() const override;
};

}
