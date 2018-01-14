//
// Created by dev on 1/13/2018.
//

#ifndef MINUSDARWIN_RUNTRACER_HPP
#define MINUSDARWIN_RUNTRACER_HPP

#include <tuple>
#include <vector>
#include <boost/chrono.hpp>
#include "Utility.hpp"

namespace MinusDarwin {
    class RunTracer {
    public:
        RunTracer();
        std::vector<Population> generations;
        std::vector<boost::chrono::duration<float> > genDurations;
    };

}

#endif //MINUSDARWIN_RUNTRACER_HPP
