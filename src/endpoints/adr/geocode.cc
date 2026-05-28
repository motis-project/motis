#include "motis/endpoints/adr/geocode.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "boost/thread/tss.hpp"

#include "utl/for_each_bit_set.h"
#include "utl/to_vec.h"

#include "fmt/format.h"

#include "net/bad_request_exception.h"

#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/translations_view.h"

#include "adr/adr.h"
#include "adr/typeahead.h"

#include "geo/latlng.h"

#include "motis/config.h"
#include "motis/endpoints/adr/filter_conv.h"
#include "motis/endpoints/adr/suggestions_to_response.h"
#include "motis/gbfs/lru_cache.h"
#include "motis/journey_to_response.h"
#include "motis/parse_location.h"
#include "motis/point_rtree.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/clasz_to_mode.h"
#include "motis/timetable/modes_to_clasz_mask.h"

namespace n = nigiri;
namespace a = adr;

namespace motis::ep {

constexpr auto const kDefaultSuggestions = 10U;

a::guess_context& get_guess_context(a::typeahead const& t, a::cache& cache) {
  auto static ctx = boost::thread_specific_ptr<a::guess_context>{};
  if (ctx.get() == nullptr || &ctx.get()->cache_ != &cache) {
    ctx.reset(new a::guess_context{cache});
  }
  ctx->resize(t);
  return *ctx;
}

std::string clean_and_normalize(std::string const& str) {
  std::string norm{adr::normalize(str)};
  std::string res;
  for (char c : norm) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == ' ' || c == '-') {
      res.push_back(std::tolower(static_cast<unsigned char>(c)));
    }
  }
  return res;
}

size_t levenshtein_distance(std::string_view s1, std::string_view s2) {
  if (s1.size() < s2.size()) {
    std::swap(s1, s2);
  }
  if (s2.empty()) return s1.size();

  std::vector<size_t> v0(s2.size() + 1);
  std::vector<size_t> v1(s2.size() + 1);

  for (size_t i = 0; i <= s2.size(); i++) {
    v0[i] = i;
  }

  for (size_t i = 0; i < s1.size(); i++) {
    v1[0] = i + 1;
    for (size_t j = 0; j < s2.size(); j++) {
      size_t cost = (s1[i] == s2[j]) ? 0 : 1;
      v1[j + 1] = std::min({v1[j] + 1, v0[j + 1] + 1, v0[j] + cost});
    }
    std::swap(v0, v1);
  }
  return v0[s2.size()];
}

constexpr std::array<std::string_view, 15> kStopWords = {
    "gare", "chemin",  "rue",  "de",  "la",  "le",      "les",    "du",
    "des",  "station", "stop", "via", "str", "strasse", "bahnhof"};

bool is_stop_word(std::string_view word) {
  for (auto const& sw : kStopWords) {
    if (word == sw) return true;
  }
  return false;
}

std::string expand_abbreviation(std::string_view word) {
  if (word == "st") return "saint";
  if (word == "ste") return "sainte";
  if (word == "pl") return "place";
  if (word == "av") return "avenue";
  if (word == "ch") return "chemin";
  if (word == "bd" || word == "blvd") return "boulevard";
  return std::string{word};
}

std::vector<std::string> get_words(std::string const& s,
                                   bool filter_stopwords = false) {
  std::vector<std::string> words;
  std::string current;
  for (char c : s) {
    if (std::isalnum(static_cast<unsigned char>(c))) {
      current.push_back(c);
    } else if (!current.empty()) {
      if (!filter_stopwords || !is_stop_word(current)) {
        words.push_back(expand_abbreviation(current));
      }
      current.clear();
    }
  }
  if (!current.empty()) {
    if (!filter_stopwords || !is_stop_word(current)) {
      words.push_back(expand_abbreviation(current));
    }
  }
  return words;
}

double compute_match_score(std::string const& norm_name,
                           std::vector<std::string> const& stop_words,
                           std::string const& norm_query,
                           std::vector<std::string> const& query_words) {

  if (query_words.empty()) {
    return 0.0;
  }

  // 1. Exact full match
  if (norm_name == norm_query) {
    return 1000.0;
  }

  // 2. Word matches
  double score = 0.0;
  size_t matched_words = 0;

  for (size_t qw_idx = 0; qw_idx < query_words.size(); ++qw_idx) {
    auto const& qw = query_words[qw_idx];
    bool word_matched = false;

    for (size_t sw_idx = 0; sw_idx < stop_words.size(); ++sw_idx) {
      auto const& sw = stop_words[sw_idx];

      if (sw == qw) {
        // Exact word match
        score += 10.0;
        // Extra bonus if first word of query matches first word of stop name
        if (qw_idx == 0 && sw_idx == 0) {
          score += 15.0;
        }
        word_matched = true;
      } else if (sw.starts_with(qw)) {
        // Prefix match of a word
        score += 5.0;
        if (qw_idx == 0 && sw_idx == 0) {
          score += 8.0;
        }
        word_matched = true;
      } else if (sw.find(qw) != std::string::npos) {
        // Substring match
        score += 2.0;
        word_matched = true;
      } else {
        // Typo tolerance (Levenshtein distance)
        size_t const dist = levenshtein_distance(sw, qw);
        if ((qw.size() > 3 && dist == 1) || (qw.size() > 6 && dist == 2)) {
          score += 3.0;
          word_matched = true;
        }
      }
    }

    if (word_matched) {
      ++matched_words;
    }
  }

  // If we matched no query words at all, then score is 0
  if (matched_words == 0) {
    return 0.0;
  }

  // 3. Completeness bonus (if all query words are matched)
  if (matched_words == query_words.size()) {
    score += 50.0;

    // Extra bonus if the stop name starts with the normalized query
    if (norm_name.starts_with(norm_query)) {
      score += 30.0;
    }
  } else {
    // Penalty for incomplete match
    score *= (static_cast<double>(matched_words) / query_words.size());
  }

  return score;
}

api::geocode_response geocode::operator()(
    boost::urls::url_view const& url) const {
  auto const params = api::geocode_params{url.params()};

  // Parse place coordinate
  auto const place = params.place_.and_then([](std::string const& s) {
    auto const parsed = parse_location(s);
    utl::verify<net::bad_request_exception>(parsed.has_value(),
                                            "could not parse place {}", s);
    return std::optional{parsed.value().pos_};
  });

  // Get required clasz mask for transit modes filtering
  auto const required_modes =
      params.mode_.transform([](std::vector<api::ModeEnum> const& modes) {
        return to_clasz_mask(modes);
      });
  auto const required_clasz = required_modes.value_or(0U);

  // Limits
  auto const config_limit = config_.get_limits().geocode_max_suggestions_;
  auto const requested_limit = params.numResults_.value_or(kDefaultSuggestions);
  utl::verify<net::bad_request_exception>(requested_limit >= 1,
                                          "limit must be >= 1");
  utl::verify<net::bad_request_exception>(
      requested_limit <= config_limit,
      "limit must be <= geocode_max_suggestions ({})", config_limit);

  // Type filters
  bool search_stops = true;
  bool search_addresses_places = true;
  if (params.type_.has_value() && !params.type_->empty()) {
    search_stops = false;
    search_addresses_places = false;
    for (auto const t : *params.type_) {
      if (t == api::LocationTypeEnum::STOP) {
        search_stops = true;
      } else if (t == api::LocationTypeEnum::ADDRESS ||
                 t == api::LocationTypeEnum::PLACE) {
        search_addresses_places = true;
      }
    }
  }

  // LRU Caching for Typeahead
  static motis::gbfs::lru_cache<std::string, std::vector<api::Match>>
      typeahead_cache_{1000};

  std::string cache_key = fmt::format(
      "{}|{}|{}|{}|{}|{}", params.text_, requested_limit,
      place.has_value() ? fmt::format("{:.4f},{:.4f}", place->lat_, place->lng_)
                        : "",
      params.placeBias_, required_clasz,
      params.language_.has_value() && !params.language_->empty()
          ? fmt::format("{}", params.language_->front())
          : "");

  if (params.type_.has_value()) {
    for (auto const t : *params.type_) {
      cache_key += fmt::format("|{}", static_cast<int>(t));
    }
  }

  if (auto cached = typeahead_cache_.get(cache_key); cached != nullptr) {
    return *cached;
  }

  // Combined results
  std::vector<api::Match> results;

  // 1. DIRECT TRANSIT STOPS SEARCH
  if (search_stops && tt_ != nullptr) {
    // Prepare query details
    std::string const norm_query = clean_and_normalize(params.text_);
    std::vector<std::string> const all_query_words =
        get_words(norm_query, false);
    std::vector<std::string> filtered_query_words = get_words(norm_query, true);
    std::vector<std::string> const& query_words =
        filtered_query_words.empty() ? all_query_words : filtered_query_words;

    if (!query_words.empty()) {
      double const place_bias = params.placeBias_;

      auto process_stop = [&](nigiri::location_idx_t const l) {
        // Skip child stops/platforms (only search top-level parents)
        if (tt_->locations_.parents_[l] != n::location_idx_t::invalid()) {
          return;
        }

        // Get stop details
        std::string const stop_name = std::string{
            tt_->get_default_translation(tt_->locations_.names_[l])};

        // Match query against stop name
        std::string const norm_name = clean_and_normalize(stop_name);
        std::vector<std::string> const stop_words = get_words(norm_name, false);

        double const text_score =
            compute_match_score(norm_name, stop_words, norm_query, query_words);
        if (text_score <= 0.0) {
          return;
        }

        // Modes & Importance
        std::optional<std::vector<api::ModeEnum>> modes;
        std::optional<double> importance;
        nigiri::routing::clasz_mask_t stop_clasz = 0U;

        if (ae_ != nullptr) {
          auto const p_idx = ae_->location_place_.at(l);
          if (p_idx != adr_extra_place_idx_t::invalid()) {
            stop_clasz = ae_->place_clasz_[p_idx];
            modes = to_modes(stop_clasz, 5);
            importance = ae_->place_importance_[p_idx];
          }
        }

        if (!modes.has_value()) {
          for (auto const r : tt_->location_routes_[l]) {
            stop_clasz |= n::routing::to_mask(tt_->route_clasz_[r]);
          }
          modes = to_modes(stop_clasz, 5);
        }

        // Filter by transit modes if required
        if (required_clasz != 0U && (stop_clasz & required_clasz) == 0U) {
          return;
        }

        // Calculate final score with distance bias and importance
        double const importance_val = importance.value_or(0.0);
        double boost = 1.0;

        geo::latlng const stop_pos = tt_->locations_.coordinates_[l];
        if (place.has_value() && stop_pos != geo::latlng{}) {
          double const distance_km = geo::distance(stop_pos, *place) / 1000.0;
          boost = 0.3 + 0.7 * (1.0 / (1.0 + (distance_km * place_bias / 5.0)));
        }

        double final_base_score = text_score / 1000.0 + importance_val * 0.05;
        if (tt_->locations_.types_[l] == nigiri::location_type::kStation) {
          final_base_score += 0.1;
        }

        double final_score = final_base_score * boost;

        // Populate premium stop metadata (level, areas, country)
        double const level = get_level(w_, pl_, matches_, l);

        std::string id;
        if (tags_ != nullptr) {
          id = tags_->id(*tt_, l);
        } else {
          id = fmt::format("stop/{}", to_idx(l));
        }

        std::vector<api::Area> api_areas;
        if (ae_ != nullptr) {
          auto const p_idx = ae_->location_place_.at(l);
          if (p_idx != adr_extra_place_idx_t::invalid()) {
            auto const t_place_idx =
                adr::place_idx_t{t_.ext_start_ + to_idx(p_idx)};
            if (t_place_idx < t_.place_areas_.size()) {
              auto const area_set_idx = t_.place_areas_[t_place_idx];
              if (area_set_idx < t_.area_sets_.size()) {
                auto const areas = t_.area_sets_[area_set_idx];
                for (auto const a : areas) {
                  auto const admin_lvl = t_.area_admin_level_[a];
                  if (admin_lvl == adr::kPostalCodeAdminLevel ||
                      admin_lvl == adr::kTimezoneAdminLevel) {
                    continue;
                  }
                  auto const area_name =
                      t_.strings_[t_.area_names_[a][adr::kDefaultLangIdx]]
                          .view();
                  api_areas.emplace_back(api::Area{
                      .name_ = std::string{area_name},
                      .adminLevel_ = static_cast<double>(to_idx(admin_lvl)),
                      .matched_ = false,
                      .unique_ = false,
                      .default_ = false});
                }
              }
            }
          }
        }

        std::vector<std::vector<double>> tokens;
        for (auto const& qw : query_words) {
          auto pos = norm_query.find(qw);
          if (pos != std::string::npos) {
            tokens.push_back(
                {static_cast<double>(pos), static_cast<double>(qw.size())});
          }
        }

        results.emplace_back(api::Match{.type_ = api::LocationTypeEnum::STOP,
                                        .category_ = std::nullopt,
                                        .tokens_ = std::move(tokens),
                                        .name_ = stop_name,
                                        .id_ = std::move(id),
                                        .lat_ = stop_pos.lat_,
                                        .lon_ = stop_pos.lng_,
                                        .level_ = level,
                                        .street_ = std::nullopt,
                                        .houseNumber_ = std::nullopt,
                                        .country_ = std::nullopt,
                                        .zip_ = std::nullopt,
                                        .tz_ = std::nullopt,
                                        .areas_ = std::move(api_areas),
                                        .score_ = final_score,
                                        .modes_ = std::move(modes),
                                        .importance_ = importance});
      };

      bool enough_local_results = false;
      if (place.has_value() && location_rtree_ != nullptr) {
        location_rtree_->in_radius(
            *place, 50000.0,
            [&](nigiri::location_idx_t const l) { process_stop(l); });
        if (results.size() >= static_cast<size_t>(requested_limit)) {
          enough_local_results = true;
        }
      }

      if (!enough_local_results) {
        results.clear();
        for (auto i = n::kNSpecialStations; i < tt_->n_locations(); ++i) {
          process_stop(n::location_idx_t{i});
        }
      }
    }
  }

  // 2. Fallback/Delegate to adr for ADDRESS and PLACE
  if (search_addresses_places) {
    auto& ctx = get_guess_context(t_, cache_);

    // Set up languages
    auto lang_indices = basic_string<a::language_idx_t>{{a::kDefaultLang}};
    if (params.language_.has_value()) {
      for (auto const& language : *params.language_) {
        auto const l_idx = t_.resolve_language(language);
        if (l_idx != a::language_idx_t::invalid()) {
          lang_indices.push_back(l_idx);
        }
      }
    }

    // Filter to ADDRESS and PLACE only
    auto const adr_type = params.type_.transform([](auto const& types) {
      auto res = std::vector<api::LocationTypeEnum>{};
      for (auto const t : types) {
        if (t == api::LocationTypeEnum::ADDRESS ||
            t == api::LocationTypeEnum::PLACE) {
          res.push_back(t);
        }
      }
      return res;
    });
    auto const filter =
        params.type_.has_value()
            ? to_filter_type(adr_type)
            : (adr::filter_type::kAddress | adr::filter_type::kPlace);

    auto const token_pos = a::get_suggestions<false>(
        t_, params.text_, static_cast<unsigned>(requested_limit), lang_indices,
        ctx, place, static_cast<float>(params.placeBias_), filter,
        std::function<bool(adr::place_idx_t)>{});

    auto adr_response =
        suggestions_to_response(t_, f_, ae_, tt_, tags_, w_, pl_, matches_,
                                lang_indices, token_pos, ctx.suggestions_);

    for (auto& match : adr_response) {
      results.push_back(std::move(match));
    }
  }

  // 3. COMBINE, SORT, LIMIT, AND RETURN
  std::sort(results.begin(), results.end(),
            [](api::Match const& a, api::Match const& b) {
              return a.score_ > b.score_;
            });

  if (results.size() > static_cast<size_t>(requested_limit)) {
    results.resize(static_cast<size_t>(requested_limit));
  }

  auto const final_results =
      std::make_shared<std::vector<api::Match>>(std::move(results));
  typeahead_cache_.try_add_or_update(cache_key,
                                     [&]() { return final_results; });

  return *final_results;
}

}  // namespace motis::ep
