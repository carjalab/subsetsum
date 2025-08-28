/*  Print formatting helper functions.
 *  
 *  :author: Dennis Aldea <dennisaldea@cmu.edu>
 */


#pragma once


#include <cinttypes>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <fmt/format.h>


namespace appfmt {

/*  Format an optional 32-bit seed as a hex string.
 *  
 *  Parameters
 *  ----------
 *  seed : std::optional<uint32_t>
 *      Seed to be formatted.
 *  
 *  Returns
 *  -------
 *  std::string
 *      - if `seed` has a value: hexadecimal representation of `seed.value()`
 *      - if `seed` does not have a value: `"(none)"`
 */
std::string seed_to_str(std::optional<uint32_t> seed) {
    if (seed) {
        return fmt::format("0x{:08X}", seed.value()); // padded for 32 bits
    } else {
        return "(none)";
    }
}

/*  Format a uBLAS matrix as a string.
 *  
 *  Parameters
 *  ----------
 *  matrix : boost::numeric::ublas::matrix<T>
 *      Matrix to be formatted.
 *  
 *  Returns
 *  -------
 *  std::string
 *      The output of `cout << matrix`.
 */
template<typename T>
std::string mtrx_to_str(boost::numeric::ublas::matrix<T> const& matrix) {
    std::stringstream outstrm;
    outstrm << matrix;
    std::string outstr = outstrm.str();
    return outstr;
}

/*  Format a uBLAS vector as a string.
 *  
 *  Parameters
 *  ----------
 *  vector : boost::numeric::ublas::vector<T>
 *      Vector to be formatted.
 *  
 *  Returns
 *  -------
 *  std::string
 *      The output of `cout << vector`.
 */
template<typename T>
std::string vctr_to_str(boost::numeric::ublas::vector<T> const& vector) {
    std::stringstream outstrm;
    outstrm << vector;
    std::string outstr = outstrm.str();
    return outstr;
}

/*  Format a uBLAS vector as a string.
 *  
 *  Parameters
 *  ----------
 *  vector : boost::numeric::ublas::vector<T, std::vector<T>>
 *      Vector to be formatted.
 *  
 *  Returns
 *  -------
 *  std::string
 *      The output of `cout << vector`.
 */
template<typename T>
std::string vctr_to_str(
        boost::numeric::ublas::vector<T, std::vector<T>> const& vector) {
    std::stringstream outstrm;
    outstrm << vector;
    std::string outstr = outstrm.str();
    return outstr;
}

}
