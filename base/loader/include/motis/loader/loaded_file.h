#pragma once

#include "boost/filesystem/path.hpp"

#include "utl/parser/buffer.h"
#include "utl/parser/cstr.h"
#include "utl/parser/file.h"

namespace motis::loader {

struct loaded_file {
  loaded_file();

  loaded_file(char const* filename, char const* str, bool convert_utf8 = false);

  loaded_file(char const* filename, std::string&& buf,
              bool convert_utf8 = false);

  explicit loaded_file(boost::filesystem::path const& p,
                       bool convert_utf8 = false);

  loaded_file(loaded_file const&) = delete;

  loaded_file(loaded_file&& o) noexcept;

  ~loaded_file();

  loaded_file& operator=(loaded_file const&) = delete;

  loaded_file& operator=(loaded_file&& o) noexcept;

  char const* name() const;

  utl::cstr content() const;

  bool empty() const;

private:
  bool contains_utf8_bom() const;

  std::string name_;
  std::string buf_;
};

}  // namespace motis::loader
