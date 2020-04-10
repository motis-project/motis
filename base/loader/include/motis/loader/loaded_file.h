#pragma once

#include "boost/filesystem/path.hpp"

#include "utl/parser/buffer.h"
#include "utl/parser/cstr.h"
#include "utl/parser/file.h"

namespace motis::loader {

struct loaded_file {
  loaded_file() = default;

  loaded_file(char const* filename, char const* str)
      : name_(filename), buf_(str) {}

  loaded_file(char const* filename, utl::buffer&& buf)
      : name_(filename), buf_(std::move(buf)) {}

  explicit loaded_file(boost::filesystem::path const& p)
      : name_(p.filename().string()),
        buf_(utl::file(p.string().c_str(), "r").content()){};

  loaded_file(loaded_file const&) = delete;

  loaded_file(loaded_file&& o) noexcept {
    name_ = std::move(o.name_);
    buf_ = std::move(o.buf_);
  }

  ~loaded_file() = default;

  loaded_file& operator=(loaded_file const&) = delete;

  loaded_file& operator=(loaded_file&& o) noexcept {
    name_ = std::move(o.name_);
    buf_ = std::move(o.buf_);
    return *this;
  }

  char const* name() const { return name_.c_str(); }

  utl::cstr content() const {
    auto const offset = contains_utf8_bom() ? 3 : 0;
    return {reinterpret_cast<char const*>(buf_.data() + offset),
            buf_.size() - offset};
  }

  bool empty() const { return buf_.size() == 0U; }

private:
  bool contains_utf8_bom() const {
    return buf_.size() >= 3 && (static_cast<int>(buf_.data()[0]) == 239 &&
                                static_cast<int>(buf_.data()[1]) == 187 &&
                                static_cast<int>(buf_.data()[2]) == 191);
  }

  std::string name_;
  utl::buffer buf_;
};

}  // namespace motis::loader
