//
// Created by dev on 1/13/2018.
//

#ifndef MINUSDARWIN_SOLVER_HPP
#define MINUSDARWIN_SOLVER_HPP

#include <cstddef>
#include <functional>
#include <boost/compute.hpp>
namespace bc = boost::compute;

namespace MinusDarwin {
    enum class GoalFunction { MaxGenerations, EpsilonReached };
    enum class CrossoverMode { Random, Best };
    struct SolverParameterSet {
        size_t popSize;  //Population size
        size_t maxGens;  //Max generations
        GoalFunction goal;  //Goal function
        CrossoverMode mode; //Crossover mode
        size_t modeDepth; //Crossover mode depth
        float epsilon;  //Epsilon for epsilon reached goal function
        float coProb;   //Crossover probability
        float diffFactor;     //Diferential factor
    };


    template <typename... ScoreFunctionArgs>
    class Solver {
    public:
        typedef std::tuple<ScoreFunctionArgs...> Agent;
        Solver(const SolverParameterSet &t_sParams,
               const std::function<float (ScoreFunctionArgs...)> &t_scoreFunction);
        Solver(const SolverParameterSet &t_sParams,
               const std::function<float (ScoreFunctionArgs...)> &t_scoreFunction,
               const bc::device &t_device);
        ~Solver() = default;
        bool useOpenCL();
        bc::device device;
        SolverParameterSet sParams;
        std::function<float (ScoreFunctionArgs...)> scoreFunction;
        //private:
    };
}

#endif //MINUSDARWIN_SOLVER_HPP
