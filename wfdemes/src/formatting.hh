/*  Print formatting helper functions.
 *  
 *  :author: Dennis Aldea <dennisaldea@cmu.edu>
 */


#pragma once


#include <optional>
#include <string>

#include <fmt/format.h>


namespace libfmt {

/*  Format an integer with zero-padding to align with a given maximum value.
 *  
 *  Parameters
 *  ----------
 *  num : int
 *      Number to be formatted.
 *  max : int
 *      Maximum number to be aligned with.
 *  
 *  Returns
 *  -------
 *  std::string
 *  
 *  Raises
 *  ------
 *  std::invalid_argument
 *      If `num` exceeds `max`.
 */
inline std::string zeropad_to_max(int num, int max) {
    const std::string PADFMT = "{:0{}}";
    // check that `num` does not exceed `max`
    if (num > max) {
        const std::string ERRFMT =
            "`num` is greater than `max`: num = {}, max = {}";
        std::string errmsg = fmt::format(ERRFMT, num, max);
        throw std::invalid_argument(errmsg);
    }
    int strwidth = std::to_string(max).length();
    return fmt::format(PADFMT, num, strwidth);
}

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
inline std::string seed_to_str(std::optional<uint32_t> seed) {
    if (seed) {
        return fmt::format("0x{:08X}", seed.value()); // padded for 32 bits
    } else {
        return "(none)";
    }
}

}
