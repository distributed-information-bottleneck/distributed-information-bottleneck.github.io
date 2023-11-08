#ifndef CCTW_H
#define CCTW_H

#include <cmath>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>
#include <iterator>
#include <random>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <cstring>

#include "cppctw.h"

namespace ctw
{
double cpp_ctw(const std::vector<char> sequence, char alphabet_size) {
    return estimate_entropy(sequence, alphabet_size);
}
}
#endif // #ifndef
