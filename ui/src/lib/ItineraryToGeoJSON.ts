import { getModeStyle } from './modeStyle';
import { type Itinerary } from './openapi/types.gen';
import polyline from 'polyline';
import { colord } from "colord";

export function itineraryToGeoJSON(i: Itinerary | null) {
  return {
    type: 'FeatureCollection',
    features: i === null ? [] : i.legs.flatMap((l) => {
      if (l.legGeometryWithLevels) {
        return l.legGeometryWithLevels.map((p) => {
          return {
            type: 'Feature',
            properties: {
              color: '#42a5f5',
              outlineColor: '#1966a4',
              level: p.from_level,
              way: p.osm_way
            },
            geometry: {
              type: 'LineString',
              coordinates: polyline.decode(p.polyline.points, 7).map(([x, y]) => [y, x])
            }
          }
        });
      } else {
        const color = `#${getModeStyle(l.mode)[1]}`;
        const outlineColor = colord(color).darken(0.2).toHex();
        return {
          type: 'Feature',
          properties: {
            outlineColor,
            color
          },
          geometry: {
            type: 'LineString',
            coordinates: polyline.decode(l.legGeometry.points, 7).map(([x, y]) => [y, x])
          }
        };
      }
    })
  };
}