//
// Created by dev on 1/13/2018.
//

#ifndef MINUSDARWIN_UTILITY_HPP
#define MINUSDARWIN_UTILITY_HPP

#include <vector>
#include <cstddef>
#include <iostream>
#include <tuple>
#include <vector>
#include <iterator>
#include <iostream>
#include <sstream>
#include <functional>
#include <utility>
#include <vector>
#include <algorithm>
#include <boost/chrono.hpp>
#include <boost/random.hpp>

namespace MinusDarwin {
    typedef std::vector<float> Agent;
    typedef std::vector<float> Scores;
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
        bool useUniformFactor;  //Uniform randomly distributed factor
    };
}

std::ostream &operator<<(std::ostream &os, const MinusDarwin::Agent &a);
#endif //MINUSDARWIN_UTILITY_HPP
