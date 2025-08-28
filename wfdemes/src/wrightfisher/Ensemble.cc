/*  Ensemble launcher for the Wright-Fisher model on demes.
 *  
 *  :author: Dennis Aldea <dennisaldea@cmu.edu>
 */


#include <wrightfisher/Ensemble.hh>

#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <magic_enum/magic_enum.hpp>

#include <wrightfisher/Simulation.hh>

#include <formatting.hh>
#include <rngfactory.hh>


namespace fs = std::filesystem;
using std::cout;
using std::ofstream;
using std::string;
using std::to_string;
using std::tuple;
using std::vector;
using fs::path;

using basemodel::EnsembleStatus;
using basemodel::SimulationResult;


namespace wrightfisher {

/*  `Ensemble` IMPLEMENTATION */

void Ensemble::create_outdir(path trajectsdir) {
    // do not overwrite existing directory
    if (fs::exists(trajectsdir)) {
        const string ERRFMT = "Output directory already exists: {}";
        string errmsg = fmt::format(ERRFMT, trajectsdir.string());
        throw std::invalid_argument(errmsg);
    }
    // `fs:filesystem_error` will be thrown if parent directory does not exist
    fs::create_directory(trajectsdir);
}

void Ensemble::make_rng_from_seedvalue(uint32_t seedvalue) {
    if (seedvalue == 0) {
        // randomize 128-bit RNG seed
        seeder_rng = make_rng<decltype(seeder_rng)>(128);
    } else {
        // repeat 32-bit "user" seed 4 times to get the 128-bit RNG seed
        seed_ = seedvalue;
        seeder_rng = make_rng<decltype(seeder_rng)>(seed.value(), 4);
    }
}

void Ensemble::print_results() const {
    for (const auto& simresult : results) {
        tuple<string, int> row_data {
            std::string{magic_enum::enum_name(simresult.outcome)},
            simresult.duration
        };
        string csvrow = fmt::format("{}\n", fmt::join(row_data, ", "));
        cout << csvrow;
    }
}

void Ensemble::write_results(path resultsfile) const {
    ofstream outfilestrm(resultsfile, std::ios_base::app); // append to file
    for (const auto& simresult : results) {
        tuple<string, int> row_data {
            std::string{magic_enum::enum_name(simresult.outcome)},
            simresult.duration
        };
        string csvrow = fmt::format("{}\n", fmt::join(row_data, ", "));
        outfilestrm << csvrow;
    }
    outfilestrm.close();
}

Ensemble::Reporter::Reporter(const Ensemble& ensmbl) {
    // get formattable representations of ensemble attributes
    string params_repr = ensmbl.params.str();
    string seed_repr = ensmbl.seed 
        ? to_string(ensmbl.seed.value()) : NoValue_repr;
    string outfile_repr = ensmbl.outfile
        ? ensmbl.outfile.value().string() : NoValue_repr;
    string outdir_repr = ensmbl.outdir
        ? ensmbl.outdir.value().string() : NoValue_repr;
    string status_repr = ensmbl.status.str();
    params = fmt::format(fmtstrs.params, params_repr);
    seed = fmt::format(fmtstrs.seed, seed_repr);
    outfile = fmt::format(fmtstrs.outfile, outfile_repr);
    ntrajs = fmt::format(fmtstrs.ntrajs, ensmbl.ntrajs);
    outdir = fmt::format(fmtstrs.outdir, outdir_repr);
    status = fmt::format(fmtstrs.status, status_repr);
}

Ensemble::Ensemble(Parameters parameterset) :
    Ensemble::Ensemble(parameterset, 0) {}

Ensemble::Ensemble(Parameters parameterset, uint32_t seedvalue) :
    params(params_),
    seed(seed_),
    outfile(outfile_),
    ntrajs(ntrajs_),
    outdir(outdir_),
    params_(parameterset),
    ntrajs_(0),
    status({0, 0, 0})
    {
        if (seedvalue == 0) { seed_.reset(); }
        make_rng_from_seedvalue(seedvalue);
    }

Ensemble::Ensemble(
        Parameters parameterset, uint32_t seedvalue,
        std::filesystem::path resultsfile) :
    params(params_),
    seed(seed_),
    outfile(outfile_),
    ntrajs(ntrajs_),
    outdir(outdir_),
    params_(parameterset),
    outfile_(resultsfile),
    ntrajs_(0),
    status({0, 0, 0})
    {
        if (seedvalue == 0) { seed_.reset(); }
        if (resultsfile == "/dev/null") { fmt::print("boop"); }
        make_rng_from_seedvalue(seedvalue);
    }

Ensemble::Ensemble(
        Parameters parameterset, uint32_t seedvalue,
        std::filesystem::path resultsfile, int numtrajects,
        std::filesystem::path trajectsdir) :
    params(params_),
    seed(seed_),
    outfile(outfile_),
    ntrajs(ntrajs_),
    outdir(outdir_),
    params_(parameterset),
    outfile_(resultsfile),
    ntrajs_(numtrajects),
    outdir_(trajectsdir),
    status({0, 0, 0})
    {
        if (seedvalue == 0) { seed_.reset(); }
        make_rng_from_seedvalue(seedvalue);
    }

const Parameters& Ensemble::get_parameters() const {
    return params;
}

EnsembleStatus Ensemble::get_status() const { return status; }

const vector<SimulationResult>& Ensemble::get_results() const {
    const string NOTENDED_ERRMSG = "Simulation has not yet ended.";
    // check that there are no unended simulations left in this ensemble
    bool ensmbl_ended = (status.pending == 0) && (status.running == 0);
    if (!ensmbl_ended) { throw std::domain_error(NOTENDED_ERRMSG); }
    return results; }

void Ensemble::run(int numruns, int maxgens) {
    status.pending = numruns;
    // set up output directory if needed
    if (outdir) { create_outdir(outdir.value()); }
    for (int runidx = 1; runidx <= numruns; ++runidx) {
        cout << fmt::format("Simulation #{}", runidx) << "\n"; //DEBUG
        // generate seed value for this simulation
        uint32_t seedvalue = seeder_rng();
        // move this simulation from pending to running
        --status.pending;
        ++status.running;
        Simulation sim = Simulation(params, seedvalue);
        if (outdir && runidx <= ntrajs) {
            // determine trajectory file path and redefine this simulation
            path odir = outdir.value();
            string traj_idxstr = libfmt::zeropad_to_max(runidx, numruns);
            string traj_fname = fmt::format(TRAJECTORY_FNAME_FMT, traj_idxstr);
            path traj_file = odir / traj_fname;
            Simulation sim = Simulation(params, seedvalue, traj_file);
        }
        // run this simulation and record the result
        sim.run(maxgens);
        results.push_back(sim.get_result());
        // move this simulation from running to ended
        --status.running;
        ++status.ended;
    }
    if (outfile) {
        write_results(outfile.value());
    } else {
        print_results();
    }
}

string Ensemble::str() const {
    Reporter reports(*this);
    const vector<string> report_vec{
        reports.params, reports.seed, reports.outfile, reports.ntrajs,
        reports.outdir, reports.status
    };
    string report = fmt::format("{}", fmt::join(report_vec, "\n"));
    return report;
}

}
