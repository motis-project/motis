#pragma once

#include <string>
#include <vector>

#include "boost/filesystem/path.hpp"

#include "flatbuffers/flatbuffers.h"

namespace flatbuffers64 {
class FlatBufferBuilder;  // NOLINT(readability-identifier-naming)
}  // namespace flatbuffers64

constexpr auto const SCHEDULE_FILE = "schedule.raw";

namespace motis::loader {

struct format_parser {
  format_parser() = default;
  format_parser(format_parser const&) = default;
  format_parser(format_parser&&) = default;
  format_parser& operator=(format_parser const&) = default;
  format_parser& operator=(format_parser&&) = default;

  virtual ~format_parser() = default;
  virtual bool applicable(boost::filesystem::path const&) = 0;
  virtual std::vector<std::string> missing_files(
      boost::filesystem::path const& path) const = 0;
  virtual void parse(boost::filesystem::path const& path,
                     flatbuffers64::FlatBufferBuilder&) = 0;
};

}  // namespace motis::loader
