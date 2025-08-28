/*  Numeric input parser.
 *  
 *  :author: Dennis Aldea <dennisaldea@cmu.edu>
 */


#pragma once


#include <limits>
#include <stdexcept>
#include <string>

#include <fmt/format.h>


/*  Parse integer value from a string.
 *  
 *  Parameters
 *  ----------
 *  str : const std::string&
 *      String representation of an integer value.
 *  
 *  Returns
 *  -------
 *  int
 *      Parsed number.
 *  
 *  Raises
 *  ------
 *  std::invalid_argument
 *      If `str` cannot be parsed as an integer number.
 *  std::out_of_range
 *      If the parsed number cannot be stored as an integer value.
 */
int parse_int(const std::string& str) {
    const std::string INVALID_ERRFMT = "Invalid integer value: {}";
    const std::string OUTRANGE_ERRFMT = "Value out of `int` range: {}";
    std::string errmsg;
    try {
        size_t pos;
        int x = std::stoi(str, &pos, 0);
        if (pos < str.size()) {
            // trailing characters after numeric input
            throw std::invalid_argument("");
        }
        return x;
    } catch (std::invalid_argument const &ex) {
        errmsg = fmt::format(INVALID_ERRFMT, str);
        throw std::invalid_argument(errmsg);
    } catch (std::out_of_range const &ex) {
        errmsg = fmt::format(OUTRANGE_ERRFMT, str);
        throw std::out_of_range(errmsg);
    }
    errmsg = fmt::format(INVALID_ERRFMT, str);
    throw std::invalid_argument(errmsg);
}

/*  Parse integer value from a character array.
 *  
 *  Parameters
 *  ----------
 *  c_str : const char*
 *      Byte string representation of an integer value.
 *  
 *  Returns
 *  -------
 *  int
 *      Parsed number.
 *  
 *  Raises
 *  ------
 *  std::invalid_argument
 *      If `c_str` cannot be parsed as an integer number.
 *  std::out_of_range
 *      If the parsed number cannot be stored as an integer value.
 */
int parse_int(const char* c_str) {
    std::string str = c_str;
    return parse_int(str);
}

/*  Parse long integer value from a string.
 *  
 *  Parameters
 *  ----------
 *  str : const std::string&
 *      String representation of a long integer value.
 *  
 *  Returns
 *  -------
 *  long
 *      Parsed number.
 *  
 *  Raises
 *  ------
 *  std::invalid_argument
 *      If `str` cannot be parsed as a integer number.
 *  std::out_of_range
 *      If the parsed number cannot be stored as a long integer value.
 */
long parse_long(const std::string& str) {
    const std::string INVALID_ERRFMT = "Invalid integer value: {}";
    const std::string OUTRANGE_ERRFMT = "Value out of `long` range: {}";
    std::string errmsg;
    try {
        size_t pos;
        long x = std::stol(str, &pos, 0);
        if (pos < str.size()) {
            // trailing characters after numeric input
            throw std::invalid_argument("");
        }
        return x;
    } catch (std::invalid_argument const &ex) {
        errmsg = fmt::format(INVALID_ERRFMT, str);
        throw std::invalid_argument(errmsg);
    } catch (std::out_of_range const &ex) {
        errmsg = fmt::format(OUTRANGE_ERRFMT, str);
        throw std::out_of_range(errmsg);
    }
    errmsg = fmt::format(INVALID_ERRFMT, str);
    throw std::invalid_argument(errmsg);
}

/*  Parse long integer value from a character array.
 *  
 *  Parameters
 *  ----------
 *  c_str : const char*
 *      Byte string representation of a long integer value.
 *  
 *  Returns
 *  -------
 *  long
 *      Parsed number.
 *  
 *  Raises
 *  ------
 *  std::invalid_argument
 *      If `c_str` cannot be parsed as a integer number.
 *  std::out_of_range
 *      If the parsed number cannot be stored as a long integer value.
 */
long parse_long(const char* c_str) {
    std::string str = c_str;
    return parse_long(str);
}

/*  Parse unsigned integer value from a string.
 *  
 *  Parameters
 *  ----------
 *  str : const std::string&
 *      String representation of an unsigned integer value.
 *  
 *  Returns
 *  -------
 *  unsigned int
 *      Parsed number.
 *  
 *  Raises
 *  ------
 *  std::invalid_argument
 *      If `str` cannot be parsed as an unsigned integer number.
 *  std::out_of_range
 *      If the parsed number cannot be stored as an unsigned integer value.
 */
unsigned int parse_uint(const std::string& str) {
    const std::string INVALID_ERRFMT = "Invalid unsigned integer value: {}";
    const std::string OUTRANGE_ERRFMT = "Value out of `unsigned int` range: {}";
    std::string errmsg;
    try {
        size_t pos;
        unsigned long x_tmp = std::stoul(str, &pos, 0);
        if (pos < str.size()) {
            // trailing characters after numeric input
            throw std::invalid_argument("");
        }
        if (x_tmp > std::numeric_limits<unsigned int>::max()) {
            // unsigned integer overflow
            throw std::out_of_range("");
        }
        unsigned int x = static_cast<unsigned int>(x_tmp);
        return x;
    } catch (std::invalid_argument const &ex) {
        errmsg = fmt::format(INVALID_ERRFMT, str);
        throw std::invalid_argument(errmsg);
    } catch (std::out_of_range const &ex) {
        errmsg = fmt::format(OUTRANGE_ERRFMT, str);
        throw std::out_of_range(errmsg);
    }
    errmsg = fmt::format(INVALID_ERRFMT, str);
    throw std::invalid_argument(errmsg);
}

/*  Parse unsigned integer value from a character array.
 *  
 *  Parameters
 *  ----------
 *  c_str : const char*
 *      Byte string representation of an unsigned integer value.
 * 
 *  Returns
 *  -------
 *  unsigned int
 *      Parsed number.
 *  
 *  Raises
 *  ------
 *  std::invalid_argument
 *      If `c_str` cannot be parsed as an unsigned integer number.
 *  std::out_of_range
 *      If the parsed number cannot be stored as an unsigned integer value.
 */
unsigned int parse_uint(const char* c_str) {
    std::string str = c_str;
    return parse_uint(str);
}

/*  Parse unsigned long integer value from a string.
 *  
 *  Parameters
 *  ----------
 *  str : const std::string&
 *      String representation of an unsigned long integer value.
 *  
 *  Returns
 *  -------
 *  unsigned long
 *      Parsed number.
 *  
 *  Raises
 *  ------
 *  std::invalid_argument
 *      If `str` cannot be parsed as a unsigned integer number.
 *  std::out_of_range
 *      If the parsed number cannot be stored as a unsigned long integer value.
 */
unsigned long parse_ulong(const std::string& str) {
    const std::string INVALID_ERRFMT = "Invalid unsigned integer value: {}";
    const std::string OUTRANGE_ERRFMT = "Value out of `unsigned long` range: {}";
    std::string errmsg;
    try {
        size_t pos;
        unsigned long x = std::stoul(str, &pos, 0);
        if (pos < str.size()) {
            // trailing characters after numeric input
            throw std::invalid_argument("");
        }
        return x;
    } catch (std::invalid_argument const &ex) {
        errmsg = fmt::format(INVALID_ERRFMT, str);
        throw std::invalid_argument(errmsg);
    } catch (std::out_of_range const &ex) {
        errmsg = fmt::format(OUTRANGE_ERRFMT, str);
        throw std::out_of_range(errmsg);
    }
    errmsg = fmt::format(INVALID_ERRFMT, str);
    throw std::invalid_argument(errmsg);
}

/*  Parse unsigned long integer value from a character array.
 *  
 *  Parameters
 *  ----------
 *  c_str : const char*
 *      Byte string representation of an unsigned long integer value.
 *  
 *  Returns
 *  -------
 *  unsigned long
 *      Parsed number.
 *  
 *  Raises
 *  ------
 *  std::invalid_argument
 *      If `c_str` cannot be parsed as an unsigned integer number.
 *  std::out_of_range
 *      If the parsed number cannot be stored as a unsigned long integer value.
 */
unsigned long parse_ulong(const char* c_str) {
    std::string str = c_str;
    return parse_ulong(str);
}

/*  Parse double-precision floating-point value from a string.
 *  
 *  Parameters
 *  ----------
 *  str : const std::string&
 *      String representation of a floating-point value.
 *  
 *  Returns
 *  -------
 *  double
 *      Parsed number.
 *  
 *  Raises
 *  ------
 *  std::invalid_argument
 *      If `str` cannot be parsed as a floating-point number.
 *  std::out_of_range
 *      If the parsed number cannot be stored as a double-precision
 *      floating-point value.
 */
double parse_double(const std::string& str) {
    const std::string INVALID_ERRFMT = "Invalid floating-point value: {}";
    const std::string OUTRANGE_ERRFMT = "Value out of `double` range: {}";
    std::string errmsg;
    try {
        size_t pos;
        double x = std::stod(str, &pos);
        if (pos < str.size()) {
            // trailing characters after numeric input
            throw std::invalid_argument("");
        }
        return x;
    } catch (std::invalid_argument const &ex) {
        errmsg = fmt::format(INVALID_ERRFMT, str);
        throw std::invalid_argument(errmsg);
    } catch (std::out_of_range const &ex) {
        errmsg = fmt::format(OUTRANGE_ERRFMT, str);
        throw std::out_of_range(errmsg);
    }
    errmsg = fmt::format(INVALID_ERRFMT, str);
    throw std::invalid_argument(errmsg);
}

/*  Parse double-precision floating-point value from a character array.
 *  
 *  Parameters
 *  ----------
 *  c_str : const char*
 *      String representation of a floating-point value.
 *  
 *  Returns
 *  -------
 *  double
 *      Parsed number.
 *  
 *  Raises
 *  ------
 *  std::invalid_argument
 *      If `c_str` cannot be parsed as a floating-point number.
 *  std::out_of_range
 *      If the parsed number cannot be stored as double-precision
 *      floating-point value.
 */
double parse_double(const char* c_str) {
    std::string str = c_str;
    return parse_double(str);
}
