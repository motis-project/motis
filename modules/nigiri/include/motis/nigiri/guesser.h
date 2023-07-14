#pragma once

#include "motis/module/message.h"

#include "nigiri/types.h"

namespace guess {
struct guesser;
}  // namespace guess

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace motis::nigiri {

struct tag_lookup;

struct guesser {
  explicit guesser(tag_lookup const&, ::nigiri::timetable const&);
  ~guesser();

  motis::module::msg_ptr guess(motis::module::msg_ptr const&);

  std::unique_ptr<guess::guesser> guess_;
  ::nigiri::timetable const& tt_;
  tag_lookup const& tags_;
  std::vector<::nigiri::location_idx_t> mapping_;
};

}  // namespace motis::nigiri