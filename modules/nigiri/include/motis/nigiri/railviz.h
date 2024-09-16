#pragma once

#include <memory>

#include "geo/box.h"

#include "nigiri/shape.h"
#include "nigiri/types.h"

#include "motis/module/message.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace motis::nigiri {

struct tag_lookup;

struct railviz {
  railviz(tag_lookup const&, ::nigiri::timetable const&,
          ::nigiri::shapes_storage&&);
  ~railviz();

  module::msg_ptr get_trains(module::msg_ptr const&) const;
  module::msg_ptr get_trips(module::msg_ptr const&) const;

  void update(std::shared_ptr<::nigiri::rt_timetable> const&) const;

  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::nigiri