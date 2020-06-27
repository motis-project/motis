#pragma once

#include <iomanip>
#include <iostream>

#include "motis/path/prepare/post/post_graph.h"

namespace motis::path {

inline void print_post_colors(post_graph const& graph, color_t const color) {
  for (auto i = 0UL; i < graph.segment_ids_.size(); ++i) {
    for (auto const& id : graph.segment_ids_.at(i)) {
      if (id.color_ != color) {
        continue;
      }

      for (auto const& sid : graph.originals_->at(i).station_ids_) {
        std::clog << sid << ".";
      }
      std::clog << std::endl;

      for (auto const& cls : graph.originals_->at(i).classes_) {
        std::clog << static_cast<service_class_t>(cls) << ",";
      }
      std::clog << std::endl;
    }
  }
}

inline void print_post_graph(post_graph const& graph) {
  std::clog << "nodes: " << graph.nodes_.size() << std::endl;
  for (auto const& node : graph.nodes_) {
    std::cout << "n: " << node->id_.osm_id_;
    for (auto ec : node->essential_) {
      std::cout << " " << ec;
    }
    std::cout << "\n";

    for (auto const& edge : node->out_) {
      std::clog << "e: " << node->id_.osm_id_ << "("
                << std::setprecision(std::numeric_limits<double>::digits10 + 1)
                << node->id_.pos_.lat_ << ","
                << std::setprecision(std::numeric_limits<double>::digits10 + 1)
                << node->id_.pos_.lng_ << ") -> " << edge.other_->id_.osm_id_
                << "("
                << std::setprecision(std::numeric_limits<double>::digits10 + 1)
                << edge.other_->id_.pos_.lat_ << ","
                << std::setprecision(std::numeric_limits<double>::digits10 + 1)
                << edge.other_->id_.pos_.lng_ << ")";

      if (edge.atomic_path_ != nullptr) {
        std::cout << " ["
                  << (edge.atomic_path_forward_
                          ? edge.atomic_path_->from_->id_.osm_id_
                          : edge.atomic_path_->to_->id_.osm_id_)
                  << "->"
                  << (edge.atomic_path_forward_
                          ? edge.atomic_path_->to_->id_.osm_id_
                          : edge.atomic_path_->from_->id_.osm_id_)
                  << " :" << edge.atomic_path_forward_ << "]";
      }

      for (auto c : edge.colors_) {
        std::cout << " " << c;
      }
      std::cout << "\n";
    }
  }
  std::clog << std::endl;
}

}  // namespace motis::path