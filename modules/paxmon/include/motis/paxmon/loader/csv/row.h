#pragma once

#include <cstdint>

#include "utl/parser/cstr.h"
#include "utl/parser/csv_range.h"

namespace motis::paxmon::loader::csv {

struct row {
  utl::csv_col<std::uint64_t, UTL_NAME("id")> id_;
  utl::csv_col<std::uint64_t, UTL_NAME("secondary_id")> secondary_id_;
  utl::csv_col<std::uint16_t, UTL_NAME("leg_idx")> leg_idx_;
  utl::csv_col<utl::cstr, UTL_NAME("leg_type")> leg_type_;
  utl::csv_col<utl::cstr, UTL_NAME("from")> from_;
  utl::csv_col<utl::cstr, UTL_NAME("to")> to_;
  utl::csv_col<std::time_t, UTL_NAME("enter")> enter_;
  utl::csv_col<std::time_t, UTL_NAME("exit")> exit_;
  utl::csv_col<utl::cstr, UTL_NAME("category")> category_;
  utl::csv_col<std::uint32_t, UTL_NAME("train_nr")> train_nr_;
  utl::csv_col<std::uint16_t, UTL_NAME("passengers")> passengers_;
};

}  // namespace motis::paxmon::loader::csv
