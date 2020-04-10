#pragma once

#include "motis/core/journey/journey.h"
#include "motis/protocol/ProblemType_generated.h"

namespace motis {

inline ProblemType problem_type_to_fbs(journey::problem_type const p) {
  switch (p) {
    case journey::problem_type::CANCELED_TRAIN:
      return ProblemType_CANCELED_TRAIN;
    case journey::problem_type::INTERCHANGE_TIME_VIOLATED:
      return ProblemType_INTERCHANGE_TIME_VIOLATED;
    default: return ProblemType_NO_PROBLEM;
  }
}

inline journey::problem_type problem_type_from_fbs(ProblemType const p) {
  switch (p) {
    case ProblemType_CANCELED_TRAIN:
      return journey::problem_type::CANCELED_TRAIN;
    case ProblemType_INTERCHANGE_TIME_VIOLATED:
      return journey::problem_type::INTERCHANGE_TIME_VIOLATED;
    default: return journey::problem_type::NO_PROBLEM;
  }
}

}  // namespace motis
