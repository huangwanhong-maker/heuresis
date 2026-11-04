// heuresis strategies: evolution strategies on ℝⁿ — the (1+1)-ES with the one-fifth rule, CMA-ES (active,
// separable, bounded) and its IPOP and BIPOP restarts — and differential evolution with SHADE.

#pragma once

#include <heuresis/strategies/cma_es.hpp>
#include <heuresis/strategies/differential_evolution.hpp>
#include <heuresis/strategies/one_plus_one.hpp>
#include <heuresis/strategies/restarts.hpp>
