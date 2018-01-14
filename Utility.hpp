//
// Created by dev on 1/13/2018.
//

#ifndef MINUSDARWIN_UTILITY_HPP
#define MINUSDARWIN_UTILITY_HPP

#include <vector>
#include <cstddef>
#include <iostream>


namespace MinusDarwin {
    typedef std::vector<float> Agent;
    typedef std::vector<Agent> Population;
    typedef std::vector<std::vector<size_t> > Neighbours;
    enum class GoalFunction { MaxGenerations, EpsilonReached };
    enum class CrossoverMode { Random, Best };
    struct SolverParameterSet {
        size_t dims;    //Number of arguments to estimate
        size_t popSize;  //Population size
        size_t maxGens;  //Max generations
        GoalFunction goal;  //Goal function
        CrossoverMode mode; //Crossover mode
        size_t modeDepth; //Crossover mode depth
        float epsilon;  //If agent score < epsilon & EpsilonReached is goal
        float coProb;   //Crossover probability
        float diffFactor;     //Diferential factor
    };
}

std::ostream &operator<<(std::ostream &os, const MinusDarwin::Agent &a);
#endif //MINUSDARWIN_UTILITY_HPP
