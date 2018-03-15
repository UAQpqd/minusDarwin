//
// Created by dev on 1/13/2018.
//


#ifndef MINUSDARWIN_SOLVER_HPP
#define MINUSDARWIN_SOLVER_HPP

#include <boost/chrono/duration.hpp>
#include "Utility.hpp"
#include "RunTracer.hpp"

#define kNeighsPerAgent(depth) (1+2*(depth))


namespace MinusDarwin {

    class Solver {
    public:
        Solver(const SolverParameterSet &t_sParams,
               const std::function<void
                       (MinusDarwin::Scores &scores,
                        const MinusDarwin::Population &population
                       )> &t_evaluatePopulation) :
                evaluatePopulation(t_evaluatePopulation), sParams(t_sParams) {
        }

        ~Solver() = default;
        Agent run(bool verbose);
        void initPopulation(Population &p);

        size_t getBestAgentId(const Scores &scores) {
            return std::min_element(scores.begin(),scores.end()) - scores.begin();
        }
        bool checkEpsilonReached(const Scores &scores, const size_t bestAgentId) {
            return scores.at(getBestAgentId(scores)) < sParams.epsilon;
        }
        void crossoverPopulation(const Population &src, Population &dst, const size_t bestAgentId) {
            auto neighbours = Neighbours(
                    sParams.popSize,
                    std::vector<size_t>(
                            kNeighsPerAgent(sParams.modeDepth), 0)
            );
            createNeighbours(neighbours, bestAgentId);
            std::vector<float> coProbs(
                    sParams.popSize * kNeighsPerAgent(sParams.modeDepth), 0.0f);
            std::vector<float> deltas(sParams.popSize);
            std::vector<std::vector<float> > randomFactors(sParams.popSize, std::vector<float>(sParams.dims,0.0f));
            boost::random::mt19937 re(boost::chrono::high_resolution_clock::now().time_since_epoch().count());
            boost::random::uniform_real_distribution<float> urd(0.0f,1.0f);
            boost::random::uniform_int_distribution<size_t> uid(0.0f, sParams.dims - 1);
            std::generate(coProbs.begin(), coProbs.end(), [&urd, &re]() { return urd(re); });
            std::generate(deltas.begin(), deltas.end(), [&uid, &re]() { return uid(re); });
            if(sParams.useUniformFactor)
                for(auto &p : randomFactors) for(auto &v : p)
                        v=urd(re);
            /* For each agent calculate the agent in Y population
             * using neighbours previously created and random numbers.
             */
            for (size_t agentId = 0; agentId < sParams.popSize; agentId++) {
                auto &agentSrc = src.at(agentId);
                auto &agentDst = dst.at(agentId);
                auto &agentNgbs = neighbours.at(agentId);
                auto agentCoProbs = std::vector<float>(
                        coProbs.begin() + agentId * kNeighsPerAgent(sParams.modeDepth),
                        coProbs.begin() + (agentId + 1) * kNeighsPerAgent(sParams.modeDepth)
                );
                const size_t delta = deltas.at(agentId);
                for (size_t paramId = 0; paramId < sParams.dims; paramId++) {
                    if (agentCoProbs.at(paramId) < sParams.coProb && delta != paramId) {
                        float newParam = 0;
                        for (size_t n = 1; n < kNeighsPerAgent(sParams.modeDepth); n++) {
                            if (n % 2 == 1)
                                newParam += sParams.diffFactor * src.at(agentNgbs.at(n)).at(paramId);
                            else
                                newParam -= sParams.diffFactor * src.at(agentNgbs.at(n)).at(paramId);
                        }
                        if(sParams.useUniformFactor)
                            newParam *= randomFactors.at(agentId).at(paramId);
                        newParam += src.at(agentNgbs.at(0)).at(paramId);
                        newParam = std::min(1.0f, std::max(0.0f, newParam));
                        agentDst.at(paramId) = newParam;
                    } else {
                        agentDst.at(paramId) = agentSrc.at(paramId);
                    }
                }
            }
        }
        void selectionPopulation(Population &main, std::vector<float> &mainScores,
                                 const Population &other, std::vector<float> &otherScores);
        void createNeighbours(Neighbours &n, const size_t bestAgentId);
        void showPopulationHead(const Population &p, const Scores &s, size_t n);
        std::string agentCSV(const Population &p,const size_t &agentId);
        SolverParameterSet sParams;
        std::function<void
                (MinusDarwin::Scores &scores,
                 const MinusDarwin::Population &population
                )> evaluatePopulation;
        RunTracer tracer;
    };
}

#endif //MINUSDARWIN_SOLVER_HPP
