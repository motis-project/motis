#include "motis/loader/hrd/model/split_service.h"

#include <cassert>
#include <algorithm>
#include <iterator>

#include "utl/verify.h"

#include "motis/loader/hrd/builder/bitfield_builder.h"
#include "motis/loader/util.h"

namespace motis::loader::hrd {

auto const all = create_uniform_bitfield<BIT_COUNT>('1');
auto const none = create_uniform_bitfield<BIT_COUNT>('0');

struct split_info {
  bitfield traffic_days_;
  int from_section_idx_, to_section_idx_;
};

struct splitter {
  explicit splitter(std::vector<bitfield> sections)
      : sections_(std::move(sections)) {}

  void check_and_remember(int start, int pos, bitfield const& b) {
    for (auto const& w : written_) {
      utl::verify((b & w.traffic_days_) == none, "invalid bitfields");
    }
    written_.push_back({b, start, pos - 1});
  }

  void write_and_remove(unsigned start, unsigned pos, bitfield current) {
    if (current != none) {
      auto not_current = (~current);
      for (unsigned i = start; i < pos; ++i) {
        sections_[i] &= not_current;
      }
      check_and_remember(start, pos, current);
    }
  }

  void split(unsigned start, unsigned pos, bitfield current) {
    if (pos == sections_.size()) {
      write_and_remove(start, pos, current);
      return;
    }

    auto intersection = current & sections_[pos];
    if (intersection == none) {
      write_and_remove(start, pos, current);
      return;
    }

    split(start, pos + 1, intersection);
    auto const diff = current & (~intersection);
    write_and_remove(start, pos, diff);
  }

  std::vector<split_info> split() {
    for (auto pos = 0UL; pos < sections_.size(); ++pos) {
      split(pos, pos, sections_[pos]);
    }
    return written_;
  }

  std::vector<bitfield> sections_;
  std::vector<split_info> written_;
};

std::vector<split_info> split(hrd_service const& s,
                              std::map<int, bitfield> const& bitfields) {
  std::vector<bitfield> section_bitfields;
  for (auto const& section : s.sections_) {
    auto it = bitfields.find(section.traffic_days_[0]);
    utl::verify(it != end(bitfields), "bitfield not found");
    section_bitfields.push_back(it->second);
  }
  return splitter(section_bitfields).split();
}

hrd_service new_service_from_split(split_info const& s,
                                   hrd_service const& origin) {
  auto number_of_stops = s.to_section_idx_ - s.from_section_idx_ + 2;
  std::vector<hrd_service::stop> stops(number_of_stops);
  std::copy(std::next(begin(origin.stops_), s.from_section_idx_),
            std::next(begin(origin.stops_), s.to_section_idx_ + 2),
            begin(stops));

  auto number_of_sections = s.to_section_idx_ - s.from_section_idx_ + 1;
  std::vector<hrd_service::section> sections(number_of_sections);
  std::copy(std::next(begin(origin.sections_), s.from_section_idx_),
            std::next(begin(origin.sections_), s.to_section_idx_ + 1),
            begin(sections));

  return {origin.origin_,
          origin.num_repetitions_,
          origin.interval_,
          stops,
          sections,
          s.traffic_days_,
          origin.initial_train_num_,
          origin.initial_admin_};
}

void expand_traffic_days(hrd_service const& service,
                         std::map<int, bitfield> const& bitfields,
                         std::vector<hrd_service>& expanded) {
  for (auto const& s : split(service, bitfields)) {
    expanded.emplace_back(new_service_from_split(s, service));
  }
}

}  // namespace motis::loader::hrd
