#pragma once

#include "motis/loader/parser.h"

namespace motis::loader::gtfs {

struct gtfs_parser : public format_parser {
  bool applicable(std::filesystem::path const&) override;
  std::vector<std::string> missing_files(
      std::filesystem::path const&) const override;
  void parse(std::filesystem::path const& root,
             flatbuffers64::FlatBufferBuilder&) override;
};

}  // namespace motis::loader::gtfs
