#pragma once

#include "utl/to_vec.h"

#include "utl/pipes/all.h"
#include "utl/pipes/remove_if.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"

#include "motis/protocol/Statistics_generated.h"

namespace motis {

struct stats_entry {
  stats_entry(std::string key, uint64_t value)
      : key_{std::move(key)}, value_{value} {}
  std::string key_;
  uint64_t value_;
};

struct stats_category {
  explicit stats_category(std::string key) : key_(std::move(key)) {}
  stats_category(std::string key, std::vector<stats_entry> entries)
      : key_(std::move(key)), entries_(std::move(entries)) {}

  std::string key_;
  std::vector<stats_entry> entries_{};
};

inline flatbuffers::Offset<Statistics> to_fbs(
    flatbuffers::FlatBufferBuilder& fbb, std::string const& category,
    std::vector<stats_entry> const& stats) {
  auto entries =
      utl::all(stats) |
      utl::remove_if([](auto&& se) { return se.value_ == 0U; }) |
      utl::transform([&](auto&& se) {
        return CreateStatisticsEntry(fbb, fbb.CreateString(se.key_), se.value_);
      }) |
      utl::vec();
  return CreateStatistics(fbb, fbb.CreateString(category),
                          fbb.CreateVectorOfSortedTables(&entries));
}

inline flatbuffers::Offset<Statistics> to_fbs(
    flatbuffers::FlatBufferBuilder& fbb, stats_category const& category) {
  return to_fbs(fbb, category.key_, category.entries_);
}

inline stats_category from_fbs(Statistics const* stats) {
  return stats_category(
      stats->category()->str(),
      utl::to_vec(*stats->entries(), [](StatisticsEntry const* entry) {
        return stats_entry{entry->name()->str(), entry->value()};
      }));
}

}  // namespace motis
