//
// Created by dev on 1/13/2018.
//

#ifndef MINUSDARWIN_UTILITY_HPP
#define MINUSDARWIN_UTILITY_HPP

#include <vector>
#include <cstddef>
#include <iostream>
#include <boost/compute.hpp>

#define dimensionShrink(v,rowSize,r,c) (\
    (v).at((r)*(rowSize)+(c))\
)

namespace bc = boost::compute;

namespace MinusDarwin {
    typedef std::vector<float> Agent;
    typedef std::vector<float> Population;
    typedef std::vector<size_t> Neighbours;
    typedef std::vector<float> Scores;
    typedef bc::vector<float> DPopulation;
    typedef bc::vector<size_t> DNeighbours;
    typedef bc::vector<float> DScores;
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
