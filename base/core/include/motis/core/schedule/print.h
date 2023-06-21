#pragma once

#include <iostream>

#include "motis/core/schedule/edges.h"
#include "motis/core/schedule/nodes.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/access/service_access.h"

namespace motis {

void print_edge(edge const& edge, std::ostream& out, int indent);
void print_node(node const& node, std::ostream& out, int indent,
                bool print_recursive);

struct indentation {
  explicit indentation(unsigned indent) : indent_(indent) {}

  friend std::ostream& operator<<(std::ostream& out, indentation const& ind) {
    for (unsigned i = 0; i < ind.indent_; ++i) out << "  ";
    return out;
  }

  unsigned indent_;
};

inline std::ostream& operator<<(std::ostream& out,
                                light_connection const& con) {
  return out << "dep=" << con.d_time_ << ", "
             << "arr=" << con.a_time_ << ", "
             << "train_nr=" << con.full_con_->con_info_->train_nr_ << ", "
             << (con.valid_ ? "" : "valid=false");
}

inline void print_node(node const& node, std::ostream& out, int indent,
                       bool print_recursive) {
  out << "node_id=" << node.id_ << " [" << node.type_str() << "]";
  if (node.is_route_node()) {
    out << " [route=" << node.route_ << "]";
  }
  out << " [#edges=" << node.edges_.size() << "]:\n";
  for (unsigned i = 0; i < node.edges_.size(); ++i) {
    out << indentation(indent + 1) << "edge[" << i << "]: ";
    print_edge(node.edges_.el_[i], out, indent + 1);

    if (print_recursive) {
      out << indentation(indent + 2);
      print_node(*node.edges_.el_[i].to_, out, indent + 2, false);
    }
  }
}

inline void print_edge(edge const& edge, std::ostream& out, int indent) {
  out << "to=" << edge.to_->id_ << " [" << edge.to_->type_str() << "], ";
  switch (edge.m_.type_) {
    case edge::ROUTE_EDGE: {
      out << "type=route_edge, "
          << "[#connections=" << edge.m_.route_edge_.conns_.size() << "]:\n";
      for (unsigned i = 0; i < edge.m_.route_edge_.conns_.size(); ++i) {
        out << indentation(indent + 1) << "connection[" << i
            << "]: " << edge.m_.route_edge_.conns_.el_[i] << "\n";
      }
      break;
    }

    case edge::FOOT_EDGE:
      out << "type=foot_edge, "
          << "duration=" << edge.m_.foot_edge_.time_cost_ << ", "
          << "transfer=" << std::boolalpha << edge.m_.foot_edge_.transfer_
          << ", "
          << (edge.m_.foot_edge_.mumo_id_ == 0
                  ? ""
                  : "mumo_id=" + std::to_string(static_cast<int>(
                                     edge.m_.foot_edge_.mumo_id_)))
          << "\n";
      break;

    case edge::THROUGH_EDGE: out << "THROUGH\n"; break;
    case edge::AFTER_TRAIN_FWD_EDGE: out << "AFTER_TRAIN_FWD\n"; break;
    case edge::AFTER_TRAIN_BWD_EDGE: out << "AFTER_TRAIN_BWD\n"; break;
    case edge::ENTER_EDGE: out << "ENTER\n"; break;
    case edge::EXIT_EDGE: out << "EXIT\n"; break;
    case edge::FWD_EDGE: out << "FWD\n"; break;
    case edge::BWD_EDGE: out << "BWD\n"; break;
    case edge::INVALID_EDGE: out << "NO_EDGE\n"; break;

    default:
      out << "UNKNOWN [type=" << static_cast<int>(edge.m_.type_) << "]\n";
  }
}

}  // namespace motis
