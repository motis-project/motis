export const toTable = (properties: Object) => {
  const table = document.createElement("table");
  table.classList.add("routing-graph", "properties");
  for (const key in properties) {
      const value = properties[key];
      const row = document.createElement("tr");
      const keyCell = document.createElement("td");
      keyCell.innerText = key;
      keyCell.className = "key";
      const valueCell = document.createElement("td");
      if (key == "osm_way_id" && value != 0) {
          const osmLink = document.createElement("a");
          osmLink.href = "https://www.openstreetmap.org/way/" + Math.abs(value);
          osmLink.target = "_blank";
          osmLink.innerText = value;
          valueCell.appendChild(osmLink);
      } else if (key == "osm_relation_id" && value != 0) {
          const osmLink = document.createElement("a");
          osmLink.href =
              "https://www.openstreetmap.org/relation/" + Math.abs(value);
          osmLink.target = "_blank";
          osmLink.innerText = value;
          valueCell.appendChild(osmLink);
      } else if (key == "osm_node_id" && value != 0) {
          const osmLink = document.createElement("a");
          osmLink.href =
              "https://www.openstreetmap.org/node/" + Math.abs(value);
          osmLink.target = "_blank";
          osmLink.innerText = value;
          valueCell.appendChild(osmLink);
      } else {
          valueCell.innerText = value;
      }
      valueCell.className = "value";
      row.appendChild(keyCell);
      row.appendChild(valueCell);
      table.appendChild(row);
  }
  return table;
}