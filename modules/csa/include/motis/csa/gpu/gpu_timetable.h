#pragma once

struct gpu_timetable;

namespace motis::csa {

struct csa_timetable;

struct gpu_timetable {
  gpu_timetable() = default;
  explicit gpu_timetable(csa_timetable&);
  ~gpu_timetable();

  gpu_timetable(gpu_timetable&&) noexcept;
  gpu_timetable& operator=(gpu_timetable&&) noexcept;

  gpu_timetable(gpu_timetable const&) = delete;
  gpu_timetable& operator=(gpu_timetable const&) = delete;

  ::gpu_timetable* ptr_{nullptr};
};

}  // namespace motis::csa
