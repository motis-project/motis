#include "motis/td_offsets.h"

#include <algorithm>
#include <optional>

#include "nigiri/footpath.h"

#include "motis/types.h"

namespace n = nigiri;

namespace motis {

bool is_same_td_state(n::routing::td_offset const& a,
                      n::routing::td_offset const& b) {
  return a.duration_ == b.duration_ &&
         a.transport_mode_id_ == b.transport_mode_id_;
}

void normalize_td_offsets(std::vector<n::routing::td_offset>& offsets) {
  if (offsets.empty()) {
    return;
  }

  // (1) lower envelope: sweep all breakpoints, keeping the fastest active
  // offer per point in time.
  std::sort(begin(offsets), end(offsets),
            [](n::routing::td_offset const& a, n::routing::td_offset const& b) {
              return a.valid_from_ < b.valid_from_;
            });

  auto active = hash_map<n::transport_mode_id_t, n::duration_t>{};
  auto envelope = std::vector<n::routing::td_offset>{};
  for (auto it = begin(offsets); it != end(offsets);) {
    auto const t = it->valid_from_;
    for (; it != end(offsets) && it->valid_from_ == t; ++it) {
      if (it->duration_ >= n::footpath::kMaxDuration) {
        active.erase(it->transport_mode_id_);
      } else {
        active[it->transport_mode_id_] = it->duration_;
      }
    }

    auto best = n::routing::td_offset{.valid_from_ = t,
                                      .duration_ = n::footpath::kMaxDuration,
                                      .transport_mode_id_ = 0U};
    for (auto const& [mode, duration] : active) {
      if (duration < best.duration_ ||
          (duration == best.duration_ && mode < best.transport_mode_id_)) {
        best.duration_ = duration;
        best.transport_mode_id_ = mode;
      }
    }

    if (envelope.empty() || !is_same_td_state(envelope.back(), best)) {
      envelope.push_back(best);
    }
  }

  // (2) FIFO repair: walking backwards, `best_arr` is the earliest arrival
  // reachable by departing at or after the current position. An active
  // window [vf, duration] is cut at `best_arr - duration`: departing later
  // than that should wait for the faster offer instead.
  auto fixed = std::vector<n::routing::td_offset>{};
  fixed.reserve(envelope.size());
  auto best_arr = std::optional<n::unixtime_t>{};
  for (auto i = envelope.size(); i-- != 0U;) {
    auto const& e = envelope[i];
    if (e.duration_ >= n::footpath::kMaxDuration) {
      fixed.push_back(e);
      continue;
    }

    auto const next_from = i + 1U < envelope.size()
                               ? envelope[i + 1U].valid_from_
                               : n::unixtime_t::max();
    if (best_arr.has_value() && *best_arr - e.duration_ <= e.valid_from_) {
      // whole window dominated by waiting for a later, faster offer
      fixed.push_back({.valid_from_ = e.valid_from_,
                       .duration_ = n::footpath::kMaxDuration,
                       .transport_mode_id_ = 0U});
      continue;
    }

    if (best_arr.has_value() && *best_arr - e.duration_ < next_from) {
      // tail of the window dominated -> end the offer early
      fixed.push_back({.valid_from_ = *best_arr - e.duration_,
                       .duration_ = n::footpath::kMaxDuration,
                       .transport_mode_id_ = 0U});
    }
    fixed.push_back(e);
    best_arr = e.valid_from_ + e.duration_;
  }
  std::reverse(begin(fixed), end(fixed));

  // Re-emit, starting with a dead window from the beginning of time so the
  // result is a complete step function, and dropping adjacent duplicates
  // introduced by the repair.
  offsets.clear();
  auto const zero = n::unixtime_t{n::i32_minutes{0}};
  if (fixed.front().valid_from_ > zero) {
    offsets.push_back({.valid_from_ = zero,
                       .duration_ = n::footpath::kMaxDuration,
                       .transport_mode_id_ = 0U});
  }
  for (auto const& e : fixed) {
    if (offsets.empty() || !is_same_td_state(offsets.back(), e)) {
      offsets.push_back(e);
    }
  }
}

}  // namespace motis
