//
// Created by dev on 1/13/2018.
//

#ifndef MINUSDARWIN_SOLVER_HPP
#define MINUSDARWIN_SOLVER_HPP

#include <cstddef>
#include <boost/compute.hpp>
namespace bc = boost::compute;

namespace MinusDarwin {

    enum class GoalFunction { MaxGenerations, EpsilonReached };
    struct SolverParameterSet {
        size_t popSize;  //Population size
        size_t maxGens;  //Max generations
        GoalFunction goal;  //Goal function
        float epsilon;  //Epsilon for epsilon reached goal function
        float coProb;   //Crossover probability
        float diffFactor;     //Diferential factor
    };
    class Solver {
    public:
        Solver(const SolverParameterSet &t_sParams);
        Solver(const SolverParameterSet &t_sParams, const bc::device &t_device);
        ~Solver() = default;
        bool useOpenCL();
        bc::device device;
        SolverParameterSet sParams;
        //private:
    };
}

#endif //MINUSDARWIN_SOLVER_HPP
