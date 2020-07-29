#include "motis/path/prepare/schedule/station_sequences.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <random>
#include <vector>

#include "cista/serialization.h"

#include "fmt/format.h"

#include "geo/box.h"

#include "utl/concat.h"
#include "utl/equal_ranges.h"
#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/to_vec.h"

#include "motis/core/common/logging.h"

#include "motis/loader/classes.h"

#include "motis/path/prepare/cista_util.h"
#include "motis/path/prepare/fbs/use_64bit_flatbuffers.h"

#include "motis/schedule-format/Schedule_generated.h"

using namespace motis::logging;

namespace motis::path {

constexpr auto const CISTA_MODE =
    cista::mode::WITH_INTEGRITY | cista::mode::WITH_VERSION;

mcd::vector<station_seq> load_station_sequences(
    motis::loader::Schedule const* sched, std::string const& prefix) {
  scoped_timer timer("loading station sequences");

  auto const& mapping = loader::class_mapping();

  std::map<motis::loader::Route const*, station_seq> seqs;
  for (auto const& service : *sched->services()) {

    auto& seq = utl::get_or_create(seqs, service->route(), [&] {
      station_seq seq;
      for (auto const& station : *service->route()->stations()) {
        seq.station_ids_.emplace_back(
            prefix.empty()
                ? station->id()->str()
                : fmt::format("{}_{}", prefix, station->id()->str()));
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

      geo::box box;
      for (auto const& c : seq.coordinates_) {
        box.extend(c);
      }
      seq.distance_ = geo::distance(box.min_, box.max_);

      return seq;
    });

    for (auto const& section : *service->sections()) {
      auto it = mapping.find(section->category()->name()->str());
      if (it != end(mapping)) {
        seq.classes_.push_back(it->second);
      }
    }
  }

  auto sequences =
      utl::to_vec(seqs, [](auto const& pair) { return pair.second; });

  mcd::vector<station_seq> result;
  utl::equal_ranges(
      sequences,
      [](auto const& lhs, auto const& rhs) {
        return lhs.station_ids_ < rhs.station_ids_;
      },
      [&](auto const& lb, auto const& ub) {
        auto& elem = *lb;

        for (auto it = std::next(lb); it != ub; ++it) {
          utl::concat(elem.classes_, it->classes_);
        }
        utl::erase_duplicates(elem.classes_);
        if (elem.classes_.empty()) {
          elem.classes_.emplace_back(service_class::OTHER);
        }

        result.emplace_back(elem);
      });

  LOG(motis::logging::info) << result.size() << " station sequences "
                            << "(was: " << sequences.size() << ")";

  return result;
}

mcd::unique_ptr<mcd::vector<station_seq>> read_station_sequences(
    std::string const& fname, cista::memory_holder& mem) {
  mcd::unique_ptr<mcd::vector<station_seq>> ptr;
  ptr.self_allocated_ = false;
#if defined(MOTIS_SCHEDULE_MODE_OFFSET) && !defined(CLANG_TIDY)
  mem = cista::buf<cista::mmap>(
      cista::mmap{fname.c_str(), cista::mmap::protection::READ});
  ptr.el_ = cista::deserialize<mcd::vector<station_seq>, CISTA_MODE>(
      std::get<cista::buf<cista::mmap>>(mem));
#elif defined(MOTIS_SCHEDULE_MODE_RAW) || defined(CLANG_TIDY)
  mem = cista::file(fname.c_str(), "r").content();
  // suppress clang-tidy false positive
  // NOLINTNEXTLINE
  ptr.el_ = cista::deserialize<mcd::vector<station_seq>, CISTA_MODE>(
      std::get<cista::buffer>(mem));
#else
#error "no ptr mode specified"
#endif
  return ptr;
}

void write_station_sequences(
    std::string const& fname,
    mcd::unique_ptr<mcd::vector<station_seq>> const& data) {
  auto writer = cista::buf<cista::mmap>(
      cista::mmap{fname.c_str(), cista::mmap::protection::WRITE});
  cista::serialize<CISTA_MODE>(writer, *data);
}

}  // namespace motis::path
