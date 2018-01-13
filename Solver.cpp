//
// Created by dev on 1/13/2018.
//

#include "Solver.hpp"

template<typename... ScoreFunctionArgs>
MinusDarwin::Solver<ScoreFunctionArgs...>::Solver(
        const SolverParameterSet &t_sParams,
        const std::function<float (ScoreFunctionArgs...)> &t_scoreFunction
) :
        Solver(t_sParams, t_scoreFunction, bc::device()) {

};

template<typename... ScoreFunctionArgs>
MinusDarwin::Solver<ScoreFunctionArgs...>::Solver(
        const SolverParameterSet &t_sParams,
        const std::function<float (ScoreFunctionArgs...)> &t_scoreFunction,
        const bc::device &t_device
) :
        scoreFunction(t_scoreFunction), sParams(t_sParams),
        device(t_device) {

}
template<typename... ScoreFunctionArgs>
bool MinusDarwin::Solver<ScoreFunctionArgs...>::useOpenCL() {
    return false;
}

template class MinusDarwin::Solver<float,float>;
template class MinusDarwin::Solver<std::vector<float> >;
