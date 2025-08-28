/* Usage help message.
 * 
 * :author: Dennis Aldea <dennisaldea@cmu.edu>
 */


#pragma once


#include <string>


/* Usage help message. */
const std::string HELPMSG =
R"(Usage
-----
wfdemes [-v] [-o <outfile>] [-n <numtrajs>] [-O <outdir>] [-s <rngseed>]
  <numsims> <maxgens> <totalpop> <mutfreq> <ecolsel> <intrsel> <migrrate>
  [<demewght> ...]

Arguments
---------
numsims : int (n ≥ 1)
    Number of simulations to run.
maxgens : int (n ≥ 0)
    Maximum number of generations per run.
totalpop : int (n ≥ 1)
    Total population size.
mutfreq : float (x > 0)
    Initial mutant frequency.
ecolsel : float (x ∈ [0, 1])
    Ecological selection coefficient.
intrsel : float (x ≥ -1)
    Intrinsic selection coefficient.
migrrate : float (x ∈ [0, 1])
    Inter-deme migration rate.
demewght ... : float (x > 0)
    List of deme weights.

Options
-------
-v, --verbose
    Enable verbose output.
-o, --outfile <outfile> : string
    Output file for ensemble results. (Default: standard output)
-n, --numtrajs <n, optional1`cout`stdoutcoustfdd::cout`trajs> : int (n ≥ 1)
    Number of simulation trajectories to write. (Default: 0)
-O, --outdir <outdir> : string
    Output directory for saved trajectories. Required if `numtrajs` given.
-s, --seed <rngseed> : int (n ∈ [0, 2^32 - 1])
    Seed for reproducible random number generation.
-h, --help
    Show this help message and exit.
)";
