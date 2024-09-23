import maplibregl from 'maplibre-gl';

const baseUrl = 'http://localhost:7999';

export class Location {
	lat!: number;
	lng!: number;
	level!: number;
	zoom!: number;
}

export class Id {
	name!: string;
	id!: string;
	src!: number;
}

export class Footpath {
	id!: Id;
	loc!: Location;
	default?: number;
	foot?: number;
	wheelchair?: number;
	wheelchair_uses_elevator?: boolean;
}

export class Footpaths {
	loc!: Location;
	id!: Id;
	footpaths!: Array<Footpath>;
}

export class RoutingQuery {
	start!: Location;
	destination!: Location;
	profile!: string;
	direction!: string;
}

export type Elevator = {
	id: string;
	desc: string;
	state: 'ACTIVE' | 'INACTIVE';
	outOfService: Array<[Date, Date]>;
};

const post = async (path: string, req: any) => {
	console.log(`FETCH ${path}: ${JSON.stringify(req)}`);
	const response = await fetch(`${baseUrl}${path}`, {
		method: 'POST',
		mode: 'cors',
		headers: {
			'Access-Control-Allow-Origin': '*',
			'Content-Type': 'application/json'
		},
		body: JSON.stringify(req)
	});
	return await response.json();
};

export const getPlatforms = async (bounds: maplibregl.LngLatBounds, level: number) => {
	return await post('/api/platforms', {
		level: level,
		waypoints: bounds.toArray().flat()
	});
};

export const getRoute = async (query: RoutingQuery) => {
	return await post('/api/route', query);
};

export const getGraph = async (bounds: maplibregl.LngLatBounds, level: number) => {
	return await post('/api/graph', {
		level: level,
		waypoints: bounds.toArray().flat()
	});
};

export const getLevels = async (bounds: maplibregl.LngLatBounds) => {
	return await post('/api/levels', {
		waypoints: bounds.toArray().flat()
	});
};

export const getMatches = async (bounds: maplibregl.LngLatBounds) => {
	return await post('/api/matches', bounds.toArray().flat());
};

export const getElevators = async (bounds: maplibregl.LngLatBounds) => {
	return await post('/api/elevators', bounds.toArray().flat());
};

export const getFootpaths = async (station: { id: string; src: number }): Promise<Footpaths> => {
	return await post('/api/footpaths', station);
};

export const updateElevator = async (elevator: Elevator) => {
	return await post('/api/update_elevator', elevator);
};
