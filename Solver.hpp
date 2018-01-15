//
// Created by dev on 1/13/2018.
//


#ifndef MINUSDARWIN_SOLVER_HPP
#define MINUSDARWIN_SOLVER_HPP

#include <functional>
#include <utility>
#include <random>
#include <vector>
#include <algorithm>
#include "RunTracer.hpp"
#include "Utility.hpp"
#include <iterator>
#include <iostream>

#define kNeighsPerAgent(depth) (1+2*(depth))


namespace MinusDarwin {

    class Solver {
    public:
        Solver(const SolverParameterSet &t_sParams,
               const std::vector<float> *t_data,
               const std::string &scoreKernelSource,
               bc::device t_device
        );
        ~Solver() = default;
        Agent run(bool verbose);
        void evaluatePopulation(
                DScores &scores,
                DPopulation &p);
        void initPopulation(Population &p);
        size_t getBestAgentId(const Scores &scores);
        size_t getBestAgentId(const DScores &scores);
        bool checkEpsilonReached(const Scores &scores, const size_t bestAgentId);
        bool checkEpsilonReached(const DScores &scores, const size_t bestAgentId);
        void crossoverPopulation(const Population &src, Population &dst, const size_t bestAgentId);
        void crossoverPopulation(const DPopulation &src, DPopulation &dst, const size_t bestAgentId);
        void selectionPopulation(Population &main, std::vector<float> &mainScores,
                                 const Population &other, std::vector<float> &otherScores);
        void createNeighbours(Neighbours &n, const size_t bestAgentId);
        void showPopulationHead(const Population &p, const std::vector<float> &s, size_t n);
        SolverParameterSet sParams;
        RunTracer tracer;
        private:
        bc::vector<float> *dData;
        bc::device device;
        bc::context ctx;
        bc::command_queue queue;
        bc::kernel scoreFunction;
    };
}

#endif //MINUSDARWIN_SOLVER_HPP
