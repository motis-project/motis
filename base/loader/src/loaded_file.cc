#include "motis/loader/loaded_file.h"

#include "utl/verify.h"

namespace motis::loader {

// Inplace version of
// https://stackoverflow.com/a/23690194
void convert_utf8_to_iso_8859_1(std::string& s) {
  auto code_point = unsigned{};
  auto out = &s[0];
  auto in = reinterpret_cast<unsigned char const*>(&s[0]);

  auto to_go = s.size();
  while (to_go != 0) {
    auto const ch = static_cast<unsigned char>(*in);

    if (ch <= 0x7f) {
      code_point = ch;
    } else if (ch <= 0xbf) {
      code_point = (code_point << 6U) | (ch & 0x3fU);
    } else if (ch <= 0xdf) {
      code_point = ch & 0x1fU;
    } else if (ch <= 0xef) {
      code_point = ch & 0x0fU;
    } else {
      code_point = ch & 0x07U;
    }

    ++in;

    if (((*in & 0xc0U) != 0x80U) && (code_point <= 0x10ffff)) {
      utl::verify(code_point <= 255, "invalid unicode");
      *out = static_cast<char>(code_point);
      ++out;
    }
    --to_go;
  }

  s.resize(out - &s[0]);
}

loaded_file::loaded_file() = default;

loaded_file::loaded_file(char const* filename, char const* str,
                         bool convert_utf8)
    : name_(filename), buf_(str) {
  if (convert_utf8) {
    convert_utf8_to_iso_8859_1(buf_);
  }
}

loaded_file::loaded_file(char const* filename, std::string&& buf,
                         bool convert_utf8)
    : name_(filename), buf_(std::move(buf)) {
  if (convert_utf8) {
    convert_utf8_to_iso_8859_1(buf_);
  }
}

loaded_file::loaded_file(boost::filesystem::path const& p, bool convert_utf8)
    : name_(p.filename().string()),
      buf_(utl::file(p.string().c_str(), "r").content_str()) {
  if (convert_utf8) {
    convert_utf8_to_iso_8859_1(buf_);
  }
}

loaded_file::loaded_file(loaded_file&& o) noexcept {
  name_ = std::move(o.name_);
  buf_ = std::move(o.buf_);
}

loaded_file::~loaded_file() = default;

loaded_file& loaded_file::operator=(loaded_file&& o) noexcept {
  name_ = std::move(o.name_);
  buf_ = std::move(o.buf_);
  return *this;
}

char const* loaded_file::name() const { return name_.c_str(); }

utl::cstr loaded_file::content() const {
  auto const offset = contains_utf8_bom() ? 3 : 0;
  return {reinterpret_cast<char const*>(buf_.data() + offset),
          buf_.size() - offset};
}

bool loaded_file::empty() const { return buf_.empty(); }

bool loaded_file::contains_utf8_bom() const {
  auto const data =
      reinterpret_cast<unsigned char const*>(buf_.data());  // NOLINT
  return buf_.size() >= 3 &&
         (static_cast<int>(data[0]) == 239 &&
          static_cast<int>(data[1]) == 187 && static_cast<int>(data[2]) == 191);
}

}  // namespace motis::loader