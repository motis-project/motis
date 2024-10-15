#include "motis/ris/zip_reader.h"

#include "miniz.h"

#include "tar/mmap_reader.h"

#include "motis/core/common/raii.h"

using namespace utl;
using tar::mmap_reader;

namespace motis::ris {

struct zip_reader::impl {
  explicit impl(char const* path)
      : mmap_{std::make_unique<mmap_reader>(path)},
        ptr_{mmap_->m_.fmap_},
        size_{mmap_->m_.size()},
        ar_{} {
    init();
  }

  impl(char const* ptr, size_t size) : ptr_{ptr}, size_{size}, ar_{} { init(); }

  impl(impl&&) = delete;
  impl& operator=(impl&&) = delete;

  impl(impl const&) = delete;
  impl& operator=(impl const&) = delete;

  ~impl() { mz_zip_reader_end(&ar_); }

  void init() {
    if (mz_zip_reader_init_mem(&ar_, ptr_, size_, 0) == 0) {
      throw std::runtime_error("invalid zip archive");
    }
    num_files_ = mz_zip_reader_get_num_files(&ar_);
  }

  size_t get_curr_file_size() {
    mz_zip_archive_file_stat file_stat{};
    if (auto const success =
            mz_zip_reader_file_stat(&ar_, curr_file_index_, &file_stat);
        success) {
      current_file_name_ = file_stat.m_filename;
      return file_stat.m_uncomp_size;
    } else {
      current_file_name_.clear();
      throw std::runtime_error("unable to parse file size");
    }
  }

  std::optional<std::string_view> read() {
    if (curr_file_index_ >= num_files_) {
      return {};
    }

    buf_.resize(get_curr_file_size());
    if (auto const success = mz_zip_reader_extract_to_mem(
            &ar_, curr_file_index_, buf_.data(), buf_.size(), 0);
        success) {
      ++curr_file_index_;
      return std::string_view{reinterpret_cast<const char*>(buf_.data()),
                              buf_.size()};
    } else {
      throw std::runtime_error("unable to read file from zip");
    }
  }

  float progress() const {
    return static_cast<float>(curr_file_index_) / num_files_;
  }

  std::string_view current_file_name() const { return current_file_name_; }

  std::unique_ptr<mmap_reader> mmap_;
  char const* ptr_{nullptr};
  size_t size_{0};

  mz_zip_archive ar_;
  size_t num_files_{0};
  size_t curr_file_index_{0};

  std::vector<unsigned char> buf_;
  std::string current_file_name_;
};

zip_reader::zip_reader(char const* path)
    : impl_{std::make_unique<impl>(path)} {}

zip_reader::zip_reader(char const* ptr, size_t size)
    : impl_{std::make_unique<impl>(ptr, size)} {}

zip_reader::~zip_reader() = default;

std::optional<std::string_view> zip_reader::read() const {
  return impl_->read();
}

float zip_reader::progress() const { return impl_->progress(); }

std::string_view zip_reader::current_file_name() const {
  return impl_->current_file_name();
}

}  // namespace motis::ris
