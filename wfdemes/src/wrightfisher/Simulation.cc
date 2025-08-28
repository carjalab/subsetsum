/*  Simulation runner for the Wright-Fisher model on demes.
 *  
 *  :author: Dennis Aldea <dennisaldea@cmu.edu>
 */


#include <wrightfisher/Simulation.hh>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <formatting.hh>
#include <rngfactory.hh>


namespace fs = std::filesystem;
using std::cout;
using std::ofstream;
using std::string;
using std::vector;
using fs::path;

namespace ublas = boost::numeric::ublas;
namespace bstrndm = boost::random;

using basemodel::SimulationOutcome;
using basemodel::SimulationStatus;

namespace wrightfisher {

/*  `Simulation` IMPLEMENTATION */

void Simulation::make_rng_from_seedvalue(uint32_t seedvalue) {
    if (seedvalue == 0) {
        // randomize 128-bit RNG seed
        rng = make_rng<decltype(rng)>(128);
    } else {
        // repeat 32-bit "user" seed 4 times to get the 128-bit RNG seed
        seed_ = seedvalue;
        rng = make_rng<decltype(rng)>(seed.value(), 4);
    }
}

void Simulation::record_generation() {
    mutant_traj_.push_back(mutant_cnts_);
}

// [TODO]: correct implementation
void Simulation::print_trajectory() const {
    path ofile = outfile.value();
    // do not overwrite existing file
    if (fs::exists(ofile)) {
        const string ERRFMT = "Output file already exists: {}";
        string errmsg = fmt::format(ERRFMT, ofile.string());
        throw std::invalid_argument(errmsg);
    }
    ofstream outfilestrm(ofile, std::ios_base::out);
    // write initial population as a comment
    string comment = fmt::format("#{}\n", fmt::join(params.deme_pops, ", "));
    outfilestrm << comment;
    for (const auto& record : mutant_traj) {
        string csvrow = fmt::format("{}\n", fmt::join(record, ", "));
        outfilestrm << csvrow;
    }
    outfilestrm.close();
}

// [TODO]: correct implementation
void Simulation::write_trajectory(path trajectpath) const {
    path ofile = outfile.value();
    // do not overwrite existing file
    if (fs::exists(ofile)) {
        const string ERRFMT = "Output file already exists: {}";
        string errmsg = fmt::format(ERRFMT, ofile.string());
        throw std::invalid_argument(errmsg);
    }
    ofstream outfilestrm(ofile, std::ios_base::out);
    // write initial population as a comment
    string comment = fmt::format("#{}\n", fmt::join(params.deme_pops, ", "));
    outfilestrm << comment;
    for (const auto& record : mutant_traj) {
        string csvrow = fmt::format("{}\n", fmt::join(record, ", "));
        outfilestrm << csvrow;
    }
    outfilestrm.close();
}

ublas::matrix<double> Simulation::calc_migration_mtrx() const {
    // value at i, j represents migration flow from deme i to deme j
    int nrows = params.num_demes;
    int ncols = params.num_demes;
    ublas::matrix<double> migrt_mtrx(nrows, ncols);
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            double migrt_val;
            if (i == j) {
                migrt_val = 1.0 - params.migrt_rate;
            } else {
                migrt_val =
                    params.migrt_rate * params.deme_pops[j]
                    / (params.total_pop - params.deme_pops[i]);
            }
            migrt_mtrx(i, j) = migrt_val;
        }
    }
    return migrt_mtrx;
}

ublas::vector<double, vector<double>> Simulation::calc_fitness_vctr() const {
    int total_muts = mutant_total();
    double mut_freq = (double) total_muts / params.total_pop;
    double tot_fitn =
        (params.intr_selec + 1)
        * calc_mutant_relfit(params.ecol_selec, mut_freq);
    double nelems = params.num_demes;
    vector<double> fitns;
    fitns.reserve(nelems);
    for (int i = 0; i < nelems; ++i) {
        double deme_mutf = (double) mutant_cnts_[i] / params.deme_pops[i];
        double deme_fitn =
            tot_fitn * deme_mutf / (tot_fitn * deme_mutf + 1.0 - deme_mutf);
        fitns.push_back(deme_fitn);
    }
    ublas::vector<double, std::vector<double>> fitn_vctr = fitns;
    return fitn_vctr;
}

Simulation::Reporter::Reporter(const Simulation& simltn) {
    // get formattable representations of simulation attributes
    string params_repr = simltn.params.str();
    string seed_repr = simltn.seed 
        ? libfmt::seed_to_str(simltn.seed.value()) : NoValue_repr;
    std::string outfile_repr = simltn.outfile
        ? simltn.outfile.value().string() : NoValue_repr;
    std::string status_repr =
        std::string{magic_enum::enum_name(simltn.status)};
    std::string outcome_repr = simltn.outcome
        ? std::string{magic_enum::enum_name(simltn.outcome.value())}
        : NoValue_repr;
    params = fmt::format(fmtstrs.params, params_repr);
    seed = fmt::format(fmtstrs.seed, seed_repr);
    outfile = fmt::format(fmtstrs.outfile, outfile_repr);
    status = fmt::format(fmtstrs.status, status_repr);
    generation = fmt::format(fmtstrs.generation, generation);
    mutant_cnts = fmt::format(fmtstrs.mutant_cnts, mutant_cnts);
    outcome = fmt::format(fmtstrs.outcome, outcome_repr);
}

Simulation::Simulation(const Parameters& parameterset) :
    Simulation::Simulation(parameterset, 0) {}

Simulation::Simulation(const Parameters& parameterset, uint32_t seedvalue) :
    params(params_),
    seed(seed_),
    outfile(outfile_),
    generation(generation_),
    mutant_cnts(mutant_cnts_),
    mutant_traj(mutant_traj_),
    params_(parameterset),
    status(SimulationStatus::Pending),
    generation_(0),
    mutant_cnts_(parameterset.init_muts) 
    {
        cout << parameterset.str(); //DEBUG
        make_rng_from_seedvalue(seedvalue);
    }

Simulation::Simulation(
        const Parameters& parameterset, uint32_t seedvalue, path trajectfile) :
    params(params_),
    seed(seed_),
    outfile(outfile_),
    generation(generation_),
    mutant_cnts(mutant_cnts_),
    mutant_traj(mutant_traj_),
    params_(parameterset),
    outfile_(trajectfile),
    status(SimulationStatus::Pending),
    generation_(0),
    mutant_cnts_(parameterset.init_muts) 
    {
        cout << parameterset.str(); //DEBUG
        make_rng_from_seedvalue(seedvalue);
    }

const Parameters& Simulation::get_parameters() const { return params; }

SimulationStatus Simulation::get_status() const { return status; }

int Simulation::get_time() const { return generation; }

const vector<int>& Simulation::get_state() const { return mutant_cnts; }

const vector<vector<int>>& Simulation::get_history() const
    { return mutant_traj; }

SimulationOutcome Simulation::get_outcome() const {
    const string NOTENDED_ERRMSG = "Simulation has not yet ended.";
    // check that simulation ended
    bool simltn_ended = (status == SimulationStatus::Ended);
    if (!simltn_ended) { throw std::domain_error(NOTENDED_ERRMSG); }
    return outcome.value();
}

int Simulation::mutant_total() const {
    return std::reduce(mutant_cnts.begin(), mutant_cnts.end());
}

void Simulation::run(int maxgens) {
    status = SimulationStatus::Running;
    // calculate migration flow matrix
    ublas::matrix<double> migrt_mtrx = calc_migration_mtrx();
    // keep track of the last recorded generation – prevents double counting
    int last_recorded_gen;
    // record initial generation
    record_generation();
    last_recorded_gen = generation;
    while (generation <= maxgens) {
        int total_muts = mutant_total();
        // check for mutant extinction
        if (total_muts == 0) {
            outcome = SimulationOutcome::Extinct;
            status = SimulationStatus::Ended;
            if (generation != last_recorded_gen) {
                record_generation();
                last_recorded_gen = generation;
            }
            break;
        }
        // check for mutant fixation
        if (total_muts == params.total_pop) {
            outcome = SimulationOutcome::Fixed;
            status = SimulationStatus::Ended;
            if (generation != last_recorded_gen) {
                record_generation();
                last_recorded_gen = generation;
            }
            break;
        }
        // advance simulation unless this would exceed `maxgen`
        if (generation+1 > maxgens) {
            outcome = SimulationOutcome::Coexistent;
            status = SimulationStatus::Ended;
            break;
        }
        ++generation_;
        // calc per-deme mutant fitness vector
        ublas::vector<double, std::vector<double>> fitn_vctr =
            calc_fitness_vctr();
        ublas::vector<double> prob_vctr = ublas::prod(migrt_mtrx, fitn_vctr);
        for (int i = 0; i < params.num_demes; ++i) {
            bstrndm::binomial_distribution binom(
                params.deme_pops[i], prob_vctr[i]);
            mutant_cnts_[i] = binom(rng);
        }
        // record state at this generation
        if (generation != last_recorded_gen) {
            record_generation();
            last_recorded_gen = generation;
        }
    }
    // save state history to outfile if specified
    if (outfile) {
        write_trajectory(outfile.value());
    }
}

string Simulation::str() const {
    Reporter reports(*this);
    const vector<string> report_vec{
        reports.params, reports.seed, reports.outfile, reports.status,
        reports.generation, reports.mutant_cnts, reports.outcome
    };
    string report = fmt::format("{}", fmt::join(report_vec, "\n"));
    return report;
}

}
