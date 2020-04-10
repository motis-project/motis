#include "motis/path/prepare/schedule/station_sequences.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <random>
#include <vector>

#include "utl/concat.h"
#include "utl/equal_ranges.h"
#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/to_vec.h"

#include "motis/core/common/logging.h"

#include "motis/loader/classes.h"

#include "motis/path/prepare/fbs/use_64bit_flatbuffers.h"

#include "motis/schedule-format/Schedule_generated.h"

using namespace motis::logging;

namespace motis::path {

std::vector<station_seq> load_station_sequences(
    motis::loader::Schedule const* sched) {
  scoped_timer timer("loading station sequences");

  auto const& mapping = loader::class_mapping();

  std::map<motis::loader::Route const*, station_seq> seqs;
  for (auto const& service : *sched->services()) {

    auto& seq = utl::get_or_create(seqs, service->route(), [&] {
      station_seq seq;
      for (auto const& station : *service->route()->stations()) {
        seq.station_ids_.emplace_back(station->id()->str());
        seq.station_names_.emplace_back(station->name()->str());

        // broken data is broken
        auto sid = station->id()->str();
        if (sid == "8704957") {  // TGV Haute Picardie
          seq.coordinates_.emplace_back(49.85911886566254, 2.8322088718414307);
        } else if (sid == "8702205") {  // Vendôme Villiers sur Loire
          seq.coordinates_.emplace_back(47.82205007381868, 1.020607352256775);
        } else {
          seq.coordinates_.emplace_back(station->lat(), station->lng());
        }
      }
      return seq;
    });

    for (auto const& section : *service->sections()) {
      auto it = mapping.find(section->category()->name()->str());
      if (it != end(mapping)) {
        seq.categories_.emplace(it->second);
      }
    }
  }

  auto sequences =
      utl::to_vec(seqs, [](auto const& pair) { return pair.second; });

  std::vector<station_seq> result;
  utl::equal_ranges(
      sequences,
      [](auto const& lhs, auto const& rhs) {
        return lhs.station_ids_ < rhs.station_ids_;
      },
      [&](auto const& lb, auto const& ub) {
        auto elem = *lb;

        for (auto it = std::next(lb); it != ub; ++it) {
          elem.categories_.insert(begin(it->categories_), end(it->categories_));
        }

        result.emplace_back(elem);
      });

  LOG(motis::logging::info) << result.size() << " station sequences "
                            << "(was: " << sequences.size() << ")";

  return result;
}

}  // namespace motis::path
