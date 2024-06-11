import maplibregl from 'maplibre-gl';

const baseUrl = 'https://osr.motis-project.de';

export class RoutingQuery {
	start!: Array<number>;
	start_level!: number;
	destination!: Array<number>;
	destination_level!: number;
	profile!: string;
	direction!: string;
}

export const getRoute = async (query: RoutingQuery) => {
	const response = await fetch(`http://localhost:8080/api/route`, {
		method: 'POST',
		mode: 'cors',
		headers: {
			'Access-Control-Allow-Origin': '*',
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({
			start: {
				lat: query.start[1],
				lng: query.start[0],
				level: query.start_level
			},
			destination: {
				lat: query.destination[1],
				lng: query.destination[0],
				level: query.destination_level
			},
			profile: query.profile,
			direction: query.direction
		})
	});
	return await response.json();
};

export const getGraph = async (bounds: maplibregl.LngLatBounds, level: number) => {
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
	const response = await fetch(`http://localhost:8080/api/matches`, {
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

export const getElevators = async (bounds: maplibregl.LngLatBounds) => {
	const response = await fetch(`http://localhost:8080/api/elevators`, {
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
