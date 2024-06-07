
import maplibregl from 'maplibre-gl';

const baseUrl = 'https://osr.motis-project.de';

export const getGraph = async (bounds: maplibregl.LngLatBounds, level: number) => {
  console.log('FETCH GRAPH');
  const response = await fetch(`${baseUrl}/api/graph`, {
    method: 'POST',
    mode: 'cors',
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      level: level,
      waypoints: bounds.toArray().flat()
    })
  });
  return await response.json();
};

export const getLevels = async (bounds: maplibregl.LngLatBounds) => {
  const response = await fetch(`${baseUrl}/api/levels`, {
    method: 'POST',
    mode: 'cors',
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      waypoints: bounds.toArray().flat()
    })
  });
  return await response.json();
};

export const getMatches = async (bounds: maplibregl.LngLatBounds) => {
  const response = await fetch(`http://localhost:8080/matches`, {
    method: 'POST',
    mode: 'cors',
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(bounds.toArray().flat())
  });
  return await response.json();
};