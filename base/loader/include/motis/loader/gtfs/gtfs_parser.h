#include "motis/loader/parser.h"

namespace motis::loader::gtfs {

struct gtfs_parser : public format_parser {
  bool applicable(boost::filesystem::path const&) override;
  std::vector<std::string> missing_files(
      boost::filesystem::path const& path) const override;
  void parse(boost::filesystem::path const& root,
             flatbuffers64::FlatBufferBuilder&) override;
};

}  // namespace motis::loader::gtfs
