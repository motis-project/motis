#include "motis/feed_labels.h"

#include "utl/verify.h"

#include "net/bad_request_exception.h"

#include "motis/tag_lookup.h"

namespace n = nigiri;

namespace motis {

feed_labels::feed_labels(config::timetable const& c, tag_lookup const& tags) {
  n_srcs_ =
      static_cast<cista::base_t<n::source_idx_t>>(tags.src_to_tag_.size());

  auto const add = [&](std::string const& name, n::source_idx_t const src) {
    auto& b = by_name_[name];
    b.resize(n_srcs_);
    b.set(src);
  };

  for (auto const& [tag, dataset] : c.datasets_) {
    auto const src = tags.get_src(tag);
    utl::verify(src != n::source_idx_t::invalid(),
                "dataset \"{}\" is not imported, please re-run the import",
                tag);
    add(tag, src);
  }

  for (auto const& [tag, dataset] : c.datasets_) {
    if (!dataset.labels_.has_value()) {
      continue;
    }
    for (auto const& label : *dataset.labels_) {
      utl::verify(!c.datasets_.contains(label),
                  "label \"{}\" of dataset \"{}\" is the same as the tag of "
                  "dataset \"{}\"",
                  label, tag, label);
      add(label, tags.get_src(tag));
    }
  }
}

feed_labels::src_set feed_labels::resolve(
    std::vector<std::string> const& names) const {
  auto b = src_set{};
  b.resize(n_srcs_);
  for (auto const& name : names) {
    auto const it = by_name_.find(name);
    utl::verify<net::bad_request_exception>(it != end(by_name_),
                                            "unknown label \"{}\"", name);
    b |= it->second;
  }
  return b;
}

feed_labels::src_set feed_labels::blocked(
    std::vector<std::string> const& include,
    std::vector<std::string> const& exclude) const {
  if (include.empty() && exclude.empty()) {
    return {};
  }

  auto included = src_set{};
  if (include.empty()) {
    included.resize(n_srcs_);
    included.one_out();
  } else {
    included = resolve(include);
  }
  included &= ~resolve(exclude);

  return ~included;
}

}  // namespace motis
