#pragma once

#include <filesystem>

namespace motis::loader::hrd {

std::filesystem::path const TEST_RESOURCES = "base/loader/test_resources/";
std::filesystem::path const SCHEDULES = TEST_RESOURCES / "hrd_schedules";

}  // namespace motis::loader::hrd
