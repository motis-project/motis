#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"

namespace flatbuffers64 {
class FlatBufferBuilder;  // NOLINT(readability-identifier-naming)
}  // namespace flatbuffers64

namespace motis::loader {

struct parser_options {
  // Set to zero to not create additional connections
  // based on proximity of stops at all.
  unsigned link_stop_distance_{100};
};

struct format_parser {
  format_parser() = default;
  format_parser(format_parser const&) = default;
  format_parser(format_parser&&) = default;
  format_parser& operator=(format_parser const&) = default;
  format_parser& operator=(format_parser&&) = default;

  virtual ~format_parser() = default;
  virtual bool applicable(std::filesystem::path const&) = 0;
  virtual std::vector<std::string> missing_files(
      std::filesystem::path const&) const = 0;
  virtual void parse(parser_options const&, std::filesystem::path const&,
                     flatbuffers64::FlatBufferBuilder&) = 0;
};

}  // namespace motis::loader
