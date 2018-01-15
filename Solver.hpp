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
               const std::function<float (Agent)> &t_scoreFunction);
        Solver(const SolverParameterSet &t_sParams,
               const std::string t_scoreFunctionKernelString,
               const std::vector<float> *t_data,
               bc::device t_device
        );
        ~Solver() = default;
        Agent run(bool verbose);
        void evaluatePopulation(
                Scores &scores,
                Population &p);
        void evaluatePopulation(
                DScores &scores,
                DPopulation &p);
        float evaluateAgent(const Agent &a);
        void initPopulation(Population &p);
        size_t getBestAgentId(const Scores &scores);
        size_t getBestAgentId(const DScores &scores);
        bool checkEpsilonReached(const Scores &scores, const size_t bestAgentId);
        bool checkEpsilonReached(const DScores &scores, const size_t bestAgentId);
        void crossoverPopulation(const Population &src, Population &dst, const size_t bestAgentId);
        void selectionPopulation(Population &main, std::vector<float> &mainScores,
                                 const Population &other, std::vector<float> &otherScores);
        void createNeighbours(Neighbours &n, const size_t bestAgentId);
        void showPopulationHead(const Population &p, const Scores &s, size_t n);
        void showPopulationHead(const DPopulation &p, const DScores &s, size_t n);
        std::string agentCSV(const Population &p,const size_t &agentId);
        SolverParameterSet sParams;
        std::function<float (Agent)> scoreFunction;
        RunTracer tracer;
        bc::device device;
        bc::context ctx;
        bc::command_queue queue;
        bc::kernel scoreFunctionKernel;
        bc::vector<float> dData;
        //private:
    };
}

#endif //MINUSDARWIN_SOLVER_HPP
