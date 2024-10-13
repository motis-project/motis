#pragma once

#include <memory>

#include "motis-api/motis-api.h"
#include "motis/fwd.h"

namespace motis {

struct railviz_static_index {
  railviz_static_index(nigiri::timetable const&);
  ~railviz_static_index();

  struct impl;
  std::unique_ptr<impl> impl_;
};

struct railviz_rt_index {
  railviz_rt_index(nigiri::timetable const&, nigiri::rt_timetable const&);
  ~railviz_rt_index();

  struct impl;
  std::unique_ptr<impl> impl_;
};

api::railviz_response get_trains(tag_lookup const&,
                                 nigiri::timetable const&,
                                 nigiri::rt_timetable const*,
                                 nigiri::shapes_storage const*,
                                 railviz_static_index::impl const&,
                                 railviz_rt_index::impl const&,
                                 api::railviz_params const&);

}  // namespace motis