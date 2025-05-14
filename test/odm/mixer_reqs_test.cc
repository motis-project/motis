#include "gtest/gtest.h"

#include "./mixer_reqs.h"

using namespace std::string_view_literals;

constexpr auto const cs0 = R"__(
departure_time, arrival_time, transfers, first_mile_mode, first_mile_duration, last_mile_mode, last_mile_duration

)__"sv;