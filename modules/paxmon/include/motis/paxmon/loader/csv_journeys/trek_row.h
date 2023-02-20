#pragma once

#include <cstdint>

#include "utl/parser/cstr.h"
#include "utl/parser/csv_range.h"

namespace motis::paxmon::loader::csv_journeys {

struct trek1_row {
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

struct trek2_row {
  utl::csv_col<std::uint64_t, UTL_NAME("REISEKETTE_ID")> id_;
  utl::csv_col<std::uint16_t, UTL_NAME("ANZAHL_PERSONEN")> passengers_;
  utl::csv_col<std::uint16_t, UTL_NAME("ANZAHL_TEILWEGE")> leg_count_;
  utl::csv_col<std::uint16_t, UTL_NAME("POSITION")> leg_idx_;
  utl::csv_col<utl::cstr, UTL_NAME("ZUGGATTUNG")> category_;
  utl::csv_col<std::uint32_t, UTL_NAME("ZUGNUMMER")> train_nr_;
  utl::csv_col<utl::cstr, UTL_NAME("EINSTIEG_SOLL_ZEIT")> enter_;
  utl::csv_col<utl::cstr, UTL_NAME("EINSTIEG_BHF_EVA_NUMMER")> from_;
  utl::csv_col<utl::cstr, UTL_NAME("AUSSTIEG_SOLL_ZEIT")> exit_;
  utl::csv_col<utl::cstr, UTL_NAME("AUSSTIEG_BHF_EVA_NUMMER")> to_;
};

}  // namespace motis::paxmon::loader::csv_journeys
