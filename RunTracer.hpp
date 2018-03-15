//
// Created by dev on 1/13/2018.
//

#ifndef MINUSDARWIN_RUNTRACER_HPP
#define MINUSDARWIN_RUNTRACER_HPP

#include "Utility.hpp"

namespace MinusDarwin {
    class RunTracer {
    public:
        RunTracer();
        std::vector<Population> generations;
        std::vector<Scores> generationsScores;
        std::vector<long long> generationsDuration;
        Agent bestAgent;
        float bestAgentScore;
    };

}

#endif //MINUSDARWIN_RUNTRACER_HPP
