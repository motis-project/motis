#include "boost/filesystem.hpp"

#include "motis/loader/hrd/parse_config.h"
#include "motis/loader/parser.h"

namespace motis::loader::hrd {

struct hrd_parser : public format_parser {
  bool applicable(boost::filesystem::path const& path) override;
  static bool applicable(boost::filesystem::path const& path, config const& c);

  std::vector<std::string> missing_files(
      boost::filesystem::path const& hrd_root) const override;

  static std::vector<std::string> missing_files(
      boost::filesystem::path const& hrd_root, config const& c);

  void parse(boost::filesystem::path const& hrd_root,
             flatbuffers64::FlatBufferBuilder&) override;
  static void parse(boost::filesystem::path const& hrd_root,
                    flatbuffers64::FlatBufferBuilder&, config const& c);
};

}  // namespace motis::loader::hrd
