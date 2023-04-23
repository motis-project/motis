#pragma once

#include <filesystem>

#include "motis/loader/hrd/parse_config.h"
#include "motis/loader/parser.h"

namespace motis::loader::hrd {

struct hrd_parser : public format_parser {
  bool applicable(std::filesystem::path const& path) override;
  static bool applicable(std::filesystem::path const& path, config const& c);

  std::vector<std::string> missing_files(
      std::filesystem::path const& hrd_root) const override;

  static std::vector<std::string> missing_files(
      std::filesystem::path const& hrd_root, config const& c);

  void parse(std::filesystem::path const& hrd_root,
             flatbuffers64::FlatBufferBuilder&) override;
  static void parse(std::filesystem::path const& hrd_root,
                    flatbuffers64::FlatBufferBuilder&, config const& c);
};

}  // namespace motis::loader::hrd
