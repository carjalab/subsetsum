/*  Command-line interface for the Wright-Fisher model on demes.
 *  
 *  Usage
 *  -----
 *  wfdemes [-v] [-o <outfile>] [-n <numtrajs>] [-O <outdir>] [-s <rngseed>]
 *    <numsims> <maxgens> <totalpop> <mutfreq> <ecolsel> <intrsel> <migrrate>
 *    [<demewght> ...]
 *  
 *  Arguments
 *  ---------
 *  numsims : int (n ≥ 1)
 *      Number of simulations to run.
 *  maxgens : int (n ≥ 0)
 *      Maximum number of generations per run.
 *  totalpop : int (n ≥ 1)
 *      Total population size.
 *  mutfreq : float (x ∈ [0, 1])
 *      Initial mutant frequency.
 *  ecolsel : float (x ∈ [0, 1])
 *      Ecological selection coefficient.
 *  intrsel : float (x ≥ -1)
 *      Intrinsic selection coefficient.
 *  migrrate : float (x ∈ [0, 1])
 *      Inter-deme migration rate.
 *  demewght ... : float (x > 0)
 *      List of deme weights.
 *  
 *  Options
 *  -------
 *  -v, --verbose
 *      Enable verbose output.
 *  -o, --outfile <outfile> : string
 *      Output file for ensemble results. (Default: standard output)
 *  -n, --numtrajs <numtrajs> : int (n ≥ 1)
 *      Number of simulation trajectories to write. (Default: 0)
 *  -O, --outdir <outdir> : string
 *      Output directory for saved trajectories. Required if `numtrajs` given.
 *  -s, --seed <rngseed> : int (n ∈ [0, 2^32 - 1])
 *      Seed for reproducible random number generation.
 *  -h, --help
 *      Show this help message and exit.
 * 
 * :author: Dennis Aldea <dennisaldea@cmu.edu>
 */


#include <getopt.h>
//#include <stdio.h>
#include <stdlib.h>

#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <formatting.hh>
#include <parsing.hh>
#include <usagehelp.hh>

#include <wrightfisher/Parameters.hh>
#include <wrightfisher/Ensemble.hh>


namespace fs = std::filesystem;
using std::cout;
using std::string;
using std::stringstream;
using std::to_string;
using std::vector;
using fs::path;

using namespace appfmt;
namespace wf = wrightfisher;

//using namespace wfsims;


/* COMMAND LINE INTERFACE */

const char* shortopts = "vo:n:O:s:h";
const struct option longopts[] = {
    {"verbose", no_argument, 0, 'v'},
    {"outfile", required_argument, 0, 'o'},
    {"numtrajs", required_argument, 0, 'n'},
    {"outdir", required_argument, 0, 'O'},
    {"seed", required_argument, 0, 's'},
    {"help", no_argument, 0, 'h'}
    // {0, 0, 0, 0}
};
const int NUM_FIXED_ARGS = 7;
/*  Argument Variables */
struct Arguments {
    /*  Verbosity flag */
    bool isverbose = false;
    /*  Ensemble outcomes path */
    std::optional<path> outfile;
    /*  Number of trajectories saved */
    std::optional<int> numtrajs;
    /*  Trajectory directory path */
    std::optional<path> outdir;
    /*  User-defined seed value */
    std::optional<uint32_t> rngseed;
    /*  Number of simulations run */
    int numsims;
    /*  Maximum generations per run */
    int maxgens;
    /*  Total population size */
    int totalpop;
    /*  Initial mutant frequency */
    double mutfreq;
    /*  Ecological selection coefficient */
    double ecolsel;
    /*  Intrinsic selection coefficient */
    double intrsel;
    /*  Inter-deme migration rate */
    double migrrate;
    /*  Deme weights */
    vector<double> demewghts;
};

struct Reporter {
    /*  Format strings used when printing arguments. */
    static inline const struct FmtStrs{
        string isverbose = "`isverbose` = {}";
        string outfile = "`outfile` = {}";
        string numtrajs = "`numtrajs` = {}";
        string outdir = "`outdir` = {}";
        string rngseed = "`rngseed` = {}";
        string numsims = "`numsims` = {}";
        string maxgens = "`maxgens` = {}";
        string totalpop = "`totalpop` = {}";
        string mutfreq = "`mutfreq` = {}";
        string ecolsel = "`ecolsel` = {}";
        string intrsel = "`intrsel` = {}";
        string migrrate = "`migrrate` = {}";
        string demewghts = "`demewghts` = {}";

        FmtStrs() {}
    } fmtstrs;
    inline static const string NoValue_repr = "(none)";

    string
        isverbose, outfile, numtrajs, outdir, rngseed, numsims, maxgens,
        totalpop, mutfreq, ecolsel, intrsel, migrrate, demewghts;
    
    Reporter(const Arguments& args) {
        // get formattable representations of argument values
        string outfile_repr =
            args.outfile ? args.outfile.value().string() : NoValue_repr;
        string numtrajs_repr =
            args.numtrajs ? to_string(args.numtrajs.value()) : NoValue_repr;
        string outdir_repr =
            args.outdir ? args.outdir.value().string() : NoValue_repr;
        string rngseed_repr =
            args.rngseed ? to_string(args.rngseed.value()) : NoValue_repr;
        isverbose = fmt::format(fmtstrs.isverbose, args.isverbose);
        outfile = fmt::format(fmtstrs.outfile, outfile_repr);
        numtrajs = fmt::format(fmtstrs.numtrajs, numtrajs_repr);
        outdir = fmt::format(fmtstrs.outdir, outdir_repr);
        rngseed = fmt::format(fmtstrs.rngseed, rngseed_repr);
        numsims = fmt::format(fmtstrs.numsims, args.numsims);
        maxgens = fmt::format(fmtstrs.maxgens, args.maxgens);
        totalpop = fmt::format(fmtstrs.totalpop, args.totalpop);
        mutfreq = fmt::format(fmtstrs.mutfreq, args.mutfreq);
        ecolsel = fmt::format(fmtstrs.ecolsel, args.ecolsel);
        intrsel = fmt::format(fmtstrs.intrsel, args.intrsel);
        migrrate = fmt::format(fmtstrs.migrrate, args.migrrate);
        demewghts = fmt::format(fmtstrs.demewghts, args.demewghts);
    }
};

void handle_exception(std::exception_ptr exptr) {
    try { if (exptr) { std::rethrow_exception(exptr); } }
    catch(const std::exception& ex) {
        // print exception message
        std::cerr << ex.what() << std::endl;
    }
}

void print_helpmsg() {
    cout << HELPMSG;
}

Arguments parse_args(int argc, char* argv[]) {
    Arguments args;
    // parse optional arguments
    int c;
    while(true) {
        int optionidx = 0;
        c = getopt_long(argc, argv, shortopts, longopts, &optionidx);
        if (c == -1) { break; }
        switch (c) {
            case 0: {
                break;
            }
            case 'v': {
                // toggle boolean
                args.isverbose = true;
                break;
            }
            case 'o': {
                // cast to path
                args.outfile = optarg;
                break;
            }
            case 'n': {
                // parse integer
                //try { args.numtrajs = parse_int(optarg); }
                //catch (...) {
                //    handle_exception(std::current_exception());
                //    exit(EXIT_FAILURE);
                //}
                args.numtrajs = parse_int(optarg);
                break;
            }
            case 'O': {
                // cast to path
                args.outdir = optarg;
                break;
            }
            case 's': {
                // parse unsigned integer
                args.rngseed = static_cast<uint32_t>(parse_uint(optarg));
                break;
            }
            case 'h': {
                print_helpmsg();
                exit(EXIT_SUCCESS);
            }
            case '?': { exit(EXIT_FAILURE); }
            default: { abort(); }
        }
    }
    // parse positional arguments
    int n_args = argc;
    int n_optargs = optind + 1;
    int n_posargs = n_args - n_optargs;
    if (!(n_args >= NUM_FIXED_ARGS)) {
        const string errmsg = "Too few arguments. See `wfdemes -h` for help.";
        throw std::invalid_argument(errmsg);
    }
    int argidx = optind;
    args.numsims = parse_int(argv[argidx++]);
    args.maxgens = parse_int(argv[argidx++]);
    args.totalpop = parse_int(argv[argidx++]);
    args.mutfreq = parse_double(argv[argidx++]);
    args.ecolsel = parse_double(argv[argidx++]);
    args.intrsel = parse_double(argv[argidx++]);
    args.migrrate = parse_double(argv[argidx++]);
    // parse remaining arguments as deme weights
    for (; argidx < n_args; ++argidx) {
        args.demewghts.push_back(parse_double(argv[argidx]));
    }
    // assume well-mixed if no deme weights are specified
    if (args.demewghts.size() == 0) { args.demewghts = {1}; }
    return args;
}

void validate_args(const Arguments& args) {
    if (args.outfile) {
        auto outfile = args.outfile.value();
        // [TODO] validate `outfile`: do not overwrite
    }
    if (args.numtrajs) {
        auto numtrajs = args.numtrajs.value();
        // validate `numtrajs`: n ≥ 0
        if (!(numtrajs >= 0)) {
            const string errfmt = "Invaild `numtrajs` value: {}";
            string errmsg = fmt::format(errfmt, numtrajs);
            throw std::invalid_argument(errmsg);
        }
        // ensure `outdir` is specified if trajectory output is desired
        if (numtrajs > 0 && !args.outdir) {
            const string errmsg = "`numtrajs` > 0: Output directory required!";
            throw std::invalid_argument(errmsg);
        }
    }
    if (args.outdir) {
        auto outdir = args.outdir.value();
        // [TODO] validate `outdir`: do not overwrite
    }
    // validate `numsims`: n > 0
    if (!(args.numsims > 0)) {
        const string errfmt = "Invaild `numsims` value: {}";
        string errmsg = fmt::format(errfmt, args.numsims);
        throw std::invalid_argument(errmsg);
    }
    // validate `maxgens`: n ≥ 0
    if (!(args.maxgens >= 0)) {
        const string errfmt = "Invaild `maxgens` value: {}";
        string errmsg = fmt::format(errfmt, args.maxgens);
        throw std::invalid_argument(errmsg);
    }
    // validate `totalpop`: n > 0
    if (!(args.totalpop > 0)) {
        const string errfmt = "Invaild `totalpop` value: {}";
        string errmsg = fmt::format(errfmt, args.totalpop);
        throw std::invalid_argument(errmsg);
    }
    // validate `mutfreq`: x ∈ [0, 1]
    if (!(args.mutfreq >= 0 && args.mutfreq <= 1)) {
        const string errfmt = "Invaild `mutfreq` value: {}";
        string errmsg = fmt::format(errfmt, args.mutfreq);
        throw std::invalid_argument(errmsg);
    }
    // validate `ecolsel`: x ∈ [0, 1]
    if (!(args.ecolsel >= 0 && args.ecolsel <= 1)) {
        const string errfmt = "Invaild `ecolsel` value: {}";
        string errmsg = fmt::format(errfmt, args.ecolsel);
        throw std::invalid_argument(errmsg);
    }
    // validate `migrrate`: x ∈ [0, 1]
    if (!(args.migrrate >= 0 && args.migrrate <= 1)) {
        const string errfmt = "Invaild `migrrate` value: {}";
        string errmsg = fmt::format(errfmt, args.migrrate);
        throw std::invalid_argument(errmsg);
    }
    // validate `demewghts`: x > 0 ∀ x
    for (auto demewght : args.demewghts) {
        if (!(demewght > 0)) {
            const string errfmt = "Invaild `demeweights` element: {}";
            string errmsg = fmt::format(errfmt, demewght);
            throw std::invalid_argument(errmsg);
        }
    }
}

void print_args(const Arguments& args) {
    Reporter reports(args);
    const vector<string> report_vec{
        reports.isverbose, reports.outfile, reports.numtrajs, reports.outdir,
        reports.rngseed, reports.numsims, reports.maxgens, reports.totalpop,
        reports.mutfreq, reports.ecolsel, reports.intrsel, reports.migrrate,
        reports.demewghts
    };
    string report = fmt::format("{}", fmt::join(report_vec, "\n"));
    cout << report << std::endl;
}

void run(const Arguments& args) {
    // run simulations (single-core – this will run on HTCondor)
    // [TODO]: Add more helpful error messages?

    wf::Parameters parameters(
        args.totalpop, args.demewghts, args.mutfreq, args.ecolsel,
        args.intrsel, args.migrrate);
    if (args.isverbose) { cout << parameters.str() << "\n"; }
    uint32_t seedvalue = args.rngseed.value_or(0);
    wf::Ensemble ensemble(parameters, seedvalue);
    if (args.outfile) {
        wf::Ensemble ensemble(parameters, seedvalue, args.outfile.value());
        if (args.numtrajs && args.numtrajs.value() == 0) {
            wf::Ensemble ensemble(
                parameters, seedvalue, args.outfile.value(),
                args.numtrajs.value(), args.outdir.value());
        }
    }
    if (args.isverbose) { cout << ensemble.str() << "\n"; }
    ensemble.run(args.numsims, args.maxgens);
    if (args.isverbose) { cout << ensemble.str() << "\n"; }
}

void launch(int argc, char* argv[]) {
    const Arguments args = parse_args(argc, argv);
    validate_args(args);
    // verbose output of command-line inputs
    if (args.isverbose) { print_args(args); }
    run(args);
}

int main(int argc, char* argv[]) {
    try { launch(argc, argv); }
    catch (...) {
        std::exception_ptr ex_ptr = std::current_exception();
        handle_exception(ex_ptr);
        exit(EXIT_FAILURE);
    }
}
