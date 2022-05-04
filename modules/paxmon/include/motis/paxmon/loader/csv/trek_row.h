#pragma once

#include <cstdint>

#include "utl/parser/cstr.h"
#include "utl/parser/csv_range.h"

namespace motis::paxmon::loader::csv {

struct trek_row {
  utl::csv_col<std::uint64_t, UTL_NAME("Id")> id_;
  utl::csv_col<std::uint16_t, UTL_NAME("AnzP")> passengers_;
  utl::csv_col<std::uint16_t, UTL_NAME("AnzTeilwege")> leg_count_;
  utl::csv_col<std::uint16_t, UTL_NAME("Position")> leg_idx_;
  utl::csv_col<utl::cstr, UTL_NAME("Zuggattung")> category_;
  utl::csv_col<std::uint32_t, UTL_NAME("ZugNr")> train_nr_;
  utl::csv_col<utl::cstr, UTL_NAME("EinZeitpunkt")> enter_;
  utl::csv_col<utl::cstr, UTL_NAME("EinHafasBhfNr")> from_;
  utl::csv_col<utl::cstr, UTL_NAME("AusZeitpunkt")> exit_;
  utl::csv_col<utl::cstr, UTL_NAME("AusHafasBhfNr")> to_;
};

}  // namespace motis::paxmon::loader::csv
