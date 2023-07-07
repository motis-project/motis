#pragma once

#include <memory>

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace motis::nigiri {

struct railviz {

  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::nigiri