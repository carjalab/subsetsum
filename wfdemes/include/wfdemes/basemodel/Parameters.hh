/*  Abstract interface for parameter set classes.
 *  
 *  :author: Dennis Aldea <dennisaldea@cmu.edu>
 */


#pragma once


#include <string>


namespace basemodel {

/*  Virtual base class for parameter set classes.
 *  
 *  Methods
 *  -------
 *  str() : std::string (virtual, const)
 *      Report all attributes in this parameter set.
 */
class Parameters {
  public:

    /*  Report all attributes in this parameter set in human-readable form.
     *  
     *  Returns
     *  -------
     *  std::string
     *      A formatted report of the attributes in this parameter set.
     */
    virtual std::string str() const = 0;
};

}
