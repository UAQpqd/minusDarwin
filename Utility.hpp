//
// Created by dev on 1/13/2018.
//

#ifndef MINUSDARWIN_UTILITY_HPP
#define MINUSDARWIN_UTILITY_HPP

#include <vector>
#include <cstddef>
#include <iostream>
#include <boost/compute.hpp>

namespace bc = boost::compute;

#define dimShrink(w,r,c) (\
    (r)*(w)+(c)\
)

namespace MinusDarwin {
    typedef std::vector<float> Agent;
    typedef std::vector<float> Scores;
    typedef std::vector<float> Population;
    typedef std::vector<size_t> Neighbours;
    typedef bc::vector<float> DAgent;
    typedef bc::vector<float> DScores;
    typedef bc::vector<float> DPopulation;
    typedef bc::vector<size_t> DNeighbours;
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
