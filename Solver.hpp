//
// Created by dev on 1/13/2018.
//


#ifndef MINUSDARWIN_SOLVER_HPP
#define MINUSDARWIN_SOLVER_HPP

#include <functional>
#include <utility>
#include <random>
#include <vector>
#include <boost/compute.hpp>
#include "RunTracer.hpp"
#include "Utility.hpp"
#include <iterator>
#include <iostream>

#define kNeighsPerAgent(depth) (1+2*(depth))

namespace bc = boost::compute;

namespace MinusDarwin {

    class Solver {
    public:
        Solver(const SolverParameterSet &t_sParams,
               const std::function<float (Agent)> &t_scoreFunction);
        Solver(const SolverParameterSet &t_sParams,
               const std::function<float (Agent)> &t_scoreFunction,
               const bc::device &t_device);
        ~Solver() = default;
        Agent run(bool verbose);
        void evaluatePopulation(
                std::vector<float> &scores,
                Population &p);
        float evaluateAgent(const Agent &a);
        void initPopulation(Population &p);
        size_t getBestAgentId(const std::vector<float> &scores);
        bool checkEpsilonReached(const std::vector<float> &scores);
        void crossoverPopulation(const Population &src, Population &dst, const size_t bestAgentId);
        void selectionPopulation(Population &main, std::vector<float> &mainScores,
                                 const Population &other, std::vector<float> &otherScores);
        void createNeighbours(Neighbours &n, const size_t bestAgentId);
        bool useOpenCL();
        void showPopulationHead(const Population &p, const std::vector<float> &s, size_t n);
        bc::device device;
        SolverParameterSet sParams;
        std::function<float (Agent)> scoreFunction;
        RunTracer tracer;
        //private:
    };
}

#endif //MINUSDARWIN_SOLVER_HPP
