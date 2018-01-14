//
// Created by dev on 1/13/2018.
//
#include "Utility.hpp"

std::ostream &operator<<(std::ostream &os, const MinusDarwin::Agent &a) {
    os << "[";
    for(size_t p = 0; p<a.size(); p++) {
        os << a.at(p);
        if(p!=a.size()-1) os << ",";
    }
    os << "]";
    return os;
}