/*  Parameter set for the Wright-Fisher model on demes.
 *  
 *  :author: Dennis Aldea <dennisaldea@cmu.edu>
 */


#include <wrightfisher/Parameters.hh>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <boost/numeric/ublas/matrix.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>


using std::cout;
using std::string;
using std::vector;

namespace ublas = boost::numeric::ublas;

namespace wrightfisher {

/*  HELPER FUNCTIONS */

vector<int> calc_deme_pops(int totalpop, vector<double> demeweights) {
    ublas::vector<double, std::vector<double>> dw_vctr (demeweights);
    /* normalize deme weights to determine deme ratios */
    double weightsum = std::accumulate(dw_vctr.begin(), dw_vctr.end(), 0.0);
    ublas::vector<double> dr_vctr = dw_vctr / weightsum;
    ublas::vector<double> dp_vctr = totalpop * dr_vctr;
    vector<int> deme_pops;
    for (const auto& demepop : dp_vctr) {
        deme_pops.push_back(round(demepop));
    }
    /* adjust demes if the resulting population differs from `totalpop`:
     *   - if total population is under target, increase the smallest deme
     *   - if total population is over target, decrease the largest deme */
    int resulttotal = std::reduce(deme_pops.begin(), deme_pops.end());
    int resulterror = resulttotal - totalpop;
    if (resulterror < 0) {
        auto mindeme = std::min_element(deme_pops.begin(), deme_pops.end());
        size_t minidx = std::distance(deme_pops.begin(), mindeme);
        deme_pops[minidx] -= resulterror;
    } else if (resulttotal > totalpop) {
        auto maxdeme = std::max_element(deme_pops.begin(), deme_pops.end());
        size_t maxidx = std::distance(deme_pops.begin(), maxdeme);
        deme_pops[maxidx] -= resulterror;
    }
    return deme_pops;
}

vector<int> calc_mutant_cnts(vector<int> demepops, double mutfreq) {
    int num_demes = size(demepops);
    vector<int> mutant_cnts(num_demes);
    for (int i = 0; i < num_demes; ++i) {
        mutant_cnts[i] = round(mutfreq * demepops[i]);
    }
    return mutant_cnts;
}

double calc_mutant_relfit(double ecolsel, double mutfreq) {
    // check that alpha is valid (0 ≤ α ≤ 1)
    if (ecolsel < 0 || ecolsel > 1) {
        const string INVALID_ERRFMT = "α out of range (0 ≤ α ≤ 1): {}";
        string errmsg = fmt::format(INVALID_ERRFMT, ecolsel);
        throw std::invalid_argument(errmsg);
    }
    if (ecolsel == 0.5) { return 1; }
    else {
        return
            pow(2*ecolsel - 1, -2)
            / (0.5*(pow(2*ecolsel - 1, -2) - 1) + mutfreq)
            - 1;
    }
}

/*  `Parameters` IMPLEMENTATION */

Parameters::Reporter::Reporter(const Parameters& params) {
    total_pop = fmt::format(fmtstrs.total_pop, params.total_pop);
    num_demes = fmt::format(fmtstrs.num_demes, params.num_demes);
    deme_pops = fmt::format(fmtstrs.deme_pops, params.deme_pops);
    init_muts = fmt::format(fmtstrs.init_muts, params.init_muts);
    ecol_selec = fmt::format(fmtstrs.ecol_selec, params.ecol_selec);
    intr_selec = fmt::format(fmtstrs.intr_selec, params.intr_selec);
    migrt_rate = fmt::format(fmtstrs.migrt_rate, params.migrt_rate);
}

Parameters::Parameters(
    int totalpop, vector<double> demeweights, double initmutfreq,
    double ecolselec, double intrselec, double migrtrate) :
        total_pop(total_pop_),
        num_demes(num_demes_),
        deme_pops(deme_pops_),
        init_muts(init_muts_),
        ecol_selec(ecol_selec_),
        intr_selec(intr_selec_),
        migrt_rate(migrt_rate_),
        total_pop_(totalpop),
        num_demes_(std::size(demeweights)),
        deme_pops_(calc_deme_pops(total_pop_, demeweights)),
        init_muts_(calc_mutant_cnts(deme_pops_, initmutfreq)),
        ecol_selec_(ecolselec),
        intr_selec_(intrselec),
        migrt_rate_(migrtrate) {}

string Parameters::str() const {
    Reporter reports(*this);
    const vector<string> report_vec{
        reports.total_pop, reports.num_demes, reports.deme_pops,
        reports.init_muts, reports.ecol_selec, reports.intr_selec,
        reports.migrt_rate
    };
    string report = fmt::format("{}", fmt::join(report_vec, "\n"));
    return report;
}

}
