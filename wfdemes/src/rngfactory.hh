/*
 *  Factory wrapper for the boost pseudo-random number generators.
 *  
 *  Warnings
 *  --------
 *  - Not cryptographically secure!
 * 
 *  References
 *  ----------
 *  .. [1] Blackman D, Vigna S. Scrambled Linear Pseudorandom Number
 *         Generators. ACM Trans Math Softw. 2021 Sep 28;47(4):1-32.
 *         doi: 10.1145/3460772.
 */


#ifndef _RNGFACTORY_HH
#define _RNGFACTORY_HH


#include <cstdint>
#include <random>

#include <boost/random.hpp>


/* Master non-deterministic RNG */
namespace {

std::random_device rd;
const int BITS_PER_SEED = 32;

template <class Seq>
boost::random::seed_seq make_seed_seq(Seq seedvalues) {
    return boost::random::seed_seq(seedvalues.begin(), seedvalues.end());
}

}

template <class RNG>
RNG make_rng(int numbits) {
    int num_seeds = numbits / BITS_PER_SEED;
    // seed the seeder with the desired number of bits
    std::vector<uint32_t> seeds;
    seeds.reserve(num_seeds);
    for (int i = 0; i < num_seeds; ++i) {
        uint32_t seed = rd();
        seeds.push_back(seed);
    }
    boost::random::seed_seq seeder = make_seed_seq(seeds);
    return RNG(seeder);
}

template <class RNG>
RNG make_rng(uint32_t seedvalue, int numrepeats) {
    // seed the seeder with the desired number of seeds
    std::vector<uint32_t> seeds;
    seeds.reserve(numrepeats);
    for (int i = 0; i < numrepeats; ++i) { seeds.push_back(seedvalue); }
    boost::random::seed_seq seeder = make_seed_seq(seeds);
    return RNG(seeder);
}


#endif
