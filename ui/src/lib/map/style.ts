import type { StyleSpecification } from 'maplibre-gl';
const colors = {
	light: {
		background: '#f8f4f0',

		water: '#99ddff',
		rail: '#a8a8a8',
		pedestrian: '#e8e7eb',
		ferryRoute: 'rgba(102, 102, 255, 0.5)',

		sport: '#d0f4be',
		sportOutline: '#b3e998',

		building: '#ded7d3',
		buildingOutline: '#cfc8c4',

		landuseComplex: '#f0e6d1',
		landuseCommercial: 'hsla(0, 60%, 87%, 0.23)',
		landuseIndustrial: '#e0e2eb',
		landuseResidential: '#ece7e4',
		landuseRetail: 'hsla(0, 60%, 87%, 0.23)',
		landuseConstruction: '#aaa69d',

		landusePark: '#b8ebad',
		landuseNatureLight: '#ddecd7',
		landuseNatureHeavy: '#a3e39c',
		landuseCemetery: '#e0e4dd',
		landuseBeach: '#fffcd3',

		indoorCorridor: '#fdfcfa',
		indoor: '#d4edff',
		indoorOutline: '#808080',
		indoorText: '#333333',

		publicTransport: 'rgba(218,140,140,0.3)',

		footway: '#fff',
		steps: '#ff4524',

		elevatorOutline: '#808080',
		elevator: '#bcf1ba',

		roadBackResidential: '#ffffff',
		roadBackNonResidential: '#ffffff',

		motorway: '#ffb366',
		motorwayLink: '#f7e06e',
		primarySecondary: '#fffbf8',
		linkTertiary: '#ffffff',
		residential: '#ffffff',
		road: '#ffffff',

		townText: '#333333',
		townTextHalo: 'white',
		text: '#333333',
		textHalo: 'white',
		citiesText: '#111111',
		citiesTextHalo: 'white',

		shield: 'shield'
	},
	dark: {
		background: '#292929',

		water: '#1f2830',
		rail: '#808080',
		pedestrian: '#292929',
		ferryRoute: 'rgba(58, 77, 139, 0.5)',

		sport: '#272525',
		sportOutline: '#272525',

		building: '#1F1F1F',
		buildingOutline: '#1A1A1A',

		landuseComplex: '#292929',
		landuseCommercial: '#292929',
		landuseIndustrial: '#353538',
		landuseResidential: '#292929',
		landuseRetail: '#292929',
		landuseConstruction: 'red',

		landusePark: '#18221f',
		landuseNatureLight: '#1e2322',
		landuseNatureHeavy: '#1a2020',
		landuseCemetery: '#202423',
		landuseBeach: '#4c4b3e',

		indoorCorridor: '#494949',
		indoor: '#1a1a1a',
		indoorOutline: '#0d0d0d',
		indoorText: '#eeeeee',

		publicTransport: 'rgba(89, 45, 45, 0.405)',

		footway: '#3D3D3D',
		steps: '#70504b',

		elevatorOutline: '#808080',
		elevator: '#3b423b',

		roadBackResidential: '#414141',
		roadBackNonResidential: '#414141',

		motorway: '#414141',
		motorwayLink: '#414141',
		primarySecondary: '#414141',
		linkTertiary: '#414141',
		residential: '#414141',
		road: '#414141',

		text: '#9a9a9a',
		textHalo: '#151515',
		townText: '#bebebe',
		townTextHalo: '#1A1A1A',
		citiesText: '#bebebe',
		citiesTextHalo: '#1A1A1A',

		shield: 'shield-dark'
	}
};

function getUrlBase(url: string): string {
	const { origin, pathname } = new URL(url);
	return origin + pathname.slice(0, pathname.lastIndexOf('/') + 1);
}

// this doesn't escape {}-parameters
function getAbsoluteUrl(base: string, relative: string): string {
	return getUrlBase(base) + relative;
}

export const getStyle = (
	theme: 'light' | 'dark',
	level: number,
	staticBaseUrl: string,
	apiBaseUrl: string
): StyleSpecification => {
	const c = colors[theme];
	return {
		version: 8,
		sources: {
			osm: {
				type: 'vector',
				tiles: [getAbsoluteUrl(apiBaseUrl, 'tiles/{z}/{x}/{y}.mvt')],
				maxzoom: 20,
				attribution: ''
			}
		},
		glyphs: getAbsoluteUrl(apiBaseUrl, 'tiles/glyphs/{fontstack}/{range}.pbf'),
		sprite: getAbsoluteUrl(staticBaseUrl, 'sprite_sdf'),
		layers: [
			{
				id: 'background',
				type: 'background',
				paint: { 'background-color': c.background }
			},
			{
				id: 'coastline',
				type: 'fill',
				source: 'osm',
				'source-layer': 'coastline',
				paint: { 'fill-color': c.water }
			},
			{
				id: 'landuse_park',
				type: 'fill',
				source: 'osm',
				'source-layer': 'landuse',
				filter: ['==', ['get', 'landuse'], 'park'],
				paint: {
					'fill-color': c.landusePark
				}
			},
			{
				id: 'landuse',
				type: 'fill',
				source: 'osm',
				'source-layer': 'landuse',
				filter: ['!in', 'landuse', 'park', 'public_transport'],
				paint: {
					'fill-color': [
						'match',
						['get', 'landuse'],
						'complex',
						c.landuseComplex,
						'commercial',
						c.landuseCommercial,
						'industrial',
						c.landuseIndustrial,
						'residential',
						c.landuseResidential,
						'retail',
						c.landuseRetail,
						'construction',
						c.landuseConstruction,

						'nature_light',
						c.landuseNatureLight,
						'nature_heavy',
						c.landuseNatureHeavy,
						'cemetery',
						c.landuseCemetery,
						'beach',
						c.landuseBeach,

						'magenta'
					]
				}
			},
			{
				id: 'water',
				type: 'fill',
				source: 'osm',
				'source-layer': 'water',
				paint: { 'fill-color': c.water }
			},
			{
				id: 'sport',
				type: 'fill',
				source: 'osm',
				'source-layer': 'sport',
				paint: {
					'fill-color': c.sport,
					'fill-outline-color': c.sportOutline
				}
			},
			{
				id: 'pedestrian',
				type: 'fill',
				source: 'osm',
				'source-layer': 'pedestrian',
				paint: { 'fill-color': c.pedestrian }
			},
			{
				id: 'waterway',
				type: 'line',
				source: 'osm',
				'source-layer': 'waterway',
				paint: { 'line-color': c.water }
			},
			{
				id: 'building',
				type: 'fill',
				source: 'osm',
				'source-layer': 'building',
				paint: {
					'fill-color': c.building,
					'fill-outline-color': c.buildingOutline,
					'fill-opacity': ['interpolate', ['linear'], ['zoom'], 14, 0, 16, 0.8]
				}
			},
			{
				id: 'indoor-corridor',
				type: 'fill',
				source: 'osm',
				'source-layer': 'indoor',
				filter: ['all', ['==', 'indoor', 'corridor'], ['==', 'level', level]],
				paint: {
					'fill-color': c.indoorCorridor,
					'fill-opacity': 0.8
				}
			},
			{
				id: 'indoor',
				type: 'fill',
				source: 'osm',
				'source-layer': 'indoor',
				filter: ['all', ['!in', 'indoor', 'corridor', 'wall', 'elevator'], ['==', 'level', level]],
				paint: {
					'fill-color': c.indoor,
					'fill-opacity': 0.8
				}
			},
			{
				id: 'indoor-outline',
				type: 'line',
				source: 'osm',
				'source-layer': 'indoor',
				filter: ['all', ['!in', 'indoor', 'corridor', 'wall', 'elevator'], ['==', 'level', level]],
				minzoom: 18,
				paint: {
					'line-color': c.indoorOutline,
					'line-width': 2
				}
			},
			{
				id: 'indoor-names',
				type: 'symbol',
				source: 'osm',
				'source-layer': 'indoor',
				minzoom: 18,
				filter: ['any', ['!has', 'level'], ['==', 'level', level]],
				layout: {
					'symbol-placement': 'point',
					'text-field': ['get', 'name'],
					'text-font': ['Noto Sans Display Regular'],
					'text-size': 12
				},
				paint: {
					'text-color': c.indoorText
				}
			},
			{
				id: 'landuse-public-transport',
				type: 'fill',
				source: 'osm',
				'source-layer': 'landuse',
				filter: [
					'all',
					['==', 'landuse', 'public_transport'],
					['any', ['!has', 'level'], ['==', 'level', level]]
				],
				paint: {
					'fill-color': c.publicTransport
				}
			},
			{
				id: 'footway',
				type: 'line',
				source: 'osm',
				'source-layer': 'road',
				filter: [
					'all',
					['in', 'highway', 'footway', 'track', 'cycleway', 'path', 'unclassified', 'service'],
					level === 0 ? ['any', ['!has', 'level'], ['==', 'level', level]] : ['==', 'level', level]
				],
				layout: {
					'line-cap': 'round'
				},
				minzoom: 14,
				paint: {
					'line-dasharray': [0.75, 1.5],
					'line-color': c.footway,
					'line-opacity': 0.5,
					'line-width': [
						'let',
						'base',
						0.4,
						[
							'interpolate',
							['linear'],
							['zoom'],
							5,
							['+', ['*', ['var', 'base'], 0.1], 1],
							9,
							['+', ['*', ['var', 'base'], 0.4], 1],
							12,
							['+', ['*', ['var', 'base'], 1], 1],
							16,
							['+', ['*', ['var', 'base'], 4], 1],
							20,
							['+', ['*', ['var', 'base'], 8], 1]
						]
					]
				}
			},
			{
				id: 'stairs-ground',
				type: 'line',
				source: 'osm',
				'source-layer': 'road',
				minzoom: 18,
				filter: [
					'all',
					['==', 'highway', 'steps'],
					level === 0
						? [
								'any',
								['!has', 'from_level'],
								['any', ['==', 'from_level', level], ['==', 'to_level', level]]
							]
						: ['any', ['==', 'from_level', level], ['==', 'to_level', level]]
				],
				paint: {
					'line-color': '#ddddddff',
					'line-width': [
						'interpolate',
						['exponential', 2],
						['zoom'],
						10,
						['*', 4, ['^', 2, -6]],
						24,
						['*', 4, ['^', 2, 8]]
					]
				}
			},
			{
				id: 'stairs-steps',
				type: 'line',
				source: 'osm',
				'source-layer': 'road',
				minzoom: 18,
				filter: [
					'all',
					['==', 'highway', 'steps'],
					level === 0
						? [
								'any',
								['!has', 'from_level'],
								['any', ['==', 'from_level', level], ['==', 'to_level', level]]
							]
						: ['any', ['==', 'from_level', level], ['==', 'to_level', level]]
				],
				paint: {
					'line-color': '#bfbfbf',
					'line-dasharray': ['literal', [0.01, 0.1]],
					'line-width': [
						'interpolate',
						['exponential', 2],
						['zoom'],
						10,
						['*', 4, ['^', 2, -6]],
						24,
						['*', 4, ['^', 2, 8]]
					]
				}
			},
			{
				id: 'stairs-rail',
				type: 'line',
				source: 'osm',
				'source-layer': 'road',
				minzoom: 18,
				filter: [
					'all',
					['==', 'highway', 'steps'],
					level === 0
						? [
								'any',
								['!has', 'from_level'],
								['any', ['==', 'from_level', level], ['==', 'to_level', level]]
							]
						: ['any', ['==', 'from_level', level], ['==', 'to_level', level]]
				],
				paint: {
					'line-color': '#808080',
					'line-width': [
						'interpolate',
						['exponential', 2],
						['zoom'],
						10,
						['*', 0.25, ['^', 2, -6]],
						24,
						['*', 0.25, ['^', 2, 8]]
					],
					'line-gap-width': [
						'interpolate',
						['exponential', 2],
						['zoom'],
						10,
						['*', 4, ['^', 2, -6]],
						24,
						['*', 4, ['^', 2, 8]]
					]
				}
			},
			{
				id: 'indoor-elevator-outline',
				type: 'circle',
				source: 'osm',
				'source-layer': 'indoor',
				minzoom: 18,
				filter: [
					'all',
					['==', 'indoor', 'elevator'],
					['<=', 'from_level', level],
					['>=', 'to_level', level]
				],
				paint: {
					'circle-color': c.elevatorOutline,
					'circle-radius': 16
				}
			},
			{
				id: 'indoor-elevator',
				type: 'circle',
				source: 'osm',
				'source-layer': 'indoor',
				minzoom: 18,
				filter: [
					'all',
					['==', 'indoor', 'elevator'],
					['<=', 'from_level', level],
					['>=', 'to_level', level]
				],
				paint: {
					'circle-color': c.elevator,
					'circle-radius': 14
				}
			},
			{
				id: 'indoor-elevator-icon',
				type: 'symbol',
				source: 'osm',
				'source-layer': 'indoor',
				minzoom: 18,
				filter: [
					'all',
					['==', 'indoor', 'elevator'],
					['<=', 'from_level', level],
					['>=', 'to_level', level]
				],
				layout: {
					'icon-image': 'elevator',
					'icon-size': 0.9
				}
			},
			{
				id: 'road_back_residential',
				type: 'line',
				source: 'osm',
				'source-layer': 'road',
				filter: ['==', 'highway', 'residential'],
				layout: {
					'line-cap': 'round'
				},
				paint: {
					'line-color': c.roadBackResidential,
					'line-width': ['interpolate', ['linear'], ['zoom'], 5, 0, 9, 0.5, 12, 1, 16, 4, 20, 20],
					'line-opacity': ['interpolate', ['linear'], ['zoom'], 12, 0.4, 15, 1]
				}
			},
			{
				id: 'road_back_non_residential',
				type: 'line',
				source: 'osm',
				'source-layer': 'road',
				filter: [
					'!in',
					'highway',
					'footway',
					'track',
					'steps',
					'cycleway',
					'path',
					'unclassified',
					'residential',
					'service'
				],
				layout: {
					'line-cap': 'round'
				},
				paint: {
					'line-color': c.roadBackNonResidential,
					'line-width': [
						'let',
						'base',
						[
							'match',
							['get', 'highway'],
							'motorway',
							4,
							['trunk', 'motorway_link'],
							3.5,
							['primary', 'secondary', 'aeroway', 'trunk_link'],
							3,
							['primary_link', 'secondary_link', 'tertiary', 'tertiary_link'],
							1.75,
							0.0
						],
						[
							'interpolate',
							['linear'],
							['zoom'],
							5,
							['+', ['*', ['var', 'base'], 0.1], 1],
							9,
							['+', ['*', ['var', 'base'], 0.4], 1],
							12,
							['+', ['*', ['var', 'base'], 1], 1],
							16,
							['+', ['*', ['var', 'base'], 4], 1],
							20,
							['+', ['*', ['var', 'base'], 8], 1]
						]
					]
				}
			},
			{
				id: 'road',
				type: 'line',
				source: 'osm',
				'source-layer': 'road',
				layout: {
					'line-cap': 'round'
				},
				filter: [
					'all',
					['has', 'ref'],
					[
						'any',
						['==', ['get', 'highway'], 'motorway'],
						['==', ['get', 'highway'], 'trunk'],
						['==', ['get', 'highway'], 'secondary'],
						['>', ['zoom'], 11]
					]
				],
				paint: {
					'line-color': [
						'match',
						['get', 'highway'],
						'motorway',
						c.motorway,
						['trunk', 'motorway_link'],
						c.motorwayLink,
						['primary', 'secondary', 'aeroway', 'trunk_link'],
						c.primarySecondary,
						['primary_link', 'secondary_link', 'tertiary', 'tertiary_link'],
						c.linkTertiary,
						'residential',
						c.residential,
						c.road
					],
					'line-width': [
						'let',
						'base',
						[
							'match',
							['get', 'highway'],
							'motorway',
							3.5,
							['trunk', 'motorway_link'],
							3,
							['primary', 'secondary', 'aeroway', 'trunk_link'],
							2.5,
							['primary_link', 'secondary_link', 'tertiary', 'tertiary_link'],
							1.75,
							'residential',
							1.5,
							0.75
						],
						[
							'interpolate',
							['linear'],
							['zoom'],
							5,
							['*', ['var', 'base'], 0.5],
							9,
							['*', ['var', 'base'], 1],
							12,
							['*', ['var', 'base'], 2],
							16,
							['*', ['var', 'base'], 2.5],
							20,
							['*', ['var', 'base'], 3]
						]
					]
				}
			},
			{
				id: 'ferry_routes',
				type: 'line',
				source: 'osm',
				'source-layer': 'ferry',
				paint: {
					'line-width': 1.5,
					'line-dasharray': [2, 3],
					'line-color': c.ferryRoute
				}
			},
			{
				id: 'rail_detail',
				type: 'line',
				source: 'osm',
				'source-layer': 'rail',
				filter: ['==', 'rail', 'detail'],
				paint: {
					'line-color': c.rail
				}
			},
			{
				id: 'rail_secondary',
				type: 'line',
				source: 'osm',
				'source-layer': 'rail',
				filter: [
					'all',
					['==', 'rail', 'secondary'],
					['any', ['!has', 'level'], ['==', 'level', level]]
				],
				paint: {
					'line-color': c.rail,
					'line-width': 1.15
				}
			},
			{
				id: 'rail_primary',
				type: 'line',
				source: 'osm',
				'source-layer': 'rail',
				filter: [
					'all',
					['==', 'rail', 'primary'],
					['any', ['!has', 'level'], ['==', 'level', level]]
				],
				paint: {
					'line-color': c.rail,
					'line-width': 1.3
				}
			},
			{
				id: 'road-ref-shield',
				type: 'symbol',
				source: 'osm',
				'source-layer': 'road',
				minzoom: 6,
				filter: [
					'all',
					['has', 'ref'],
					[
						'any',
						['==', ['get', 'highway'], 'motorway'],
						['==', ['get', 'highway'], 'trunk'],
						['==', ['get', 'highway'], 'secondary'],
						['>', ['zoom'], 11]
					]
				],
				layout: {
					'symbol-placement': 'line',
					'text-field': ['get', 'ref'],
					'text-font': ['Noto Sans Display Regular'],
					'text-size': ['case', ['==', ['get', 'highway'], 'motorway'], 11, 10],
					'text-justify': 'center',
					'text-rotation-alignment': 'viewport',
					'text-pitch-alignment': 'viewport',
					'icon-image': c.shield,
					'icon-text-fit': 'both',
					'icon-text-fit-padding': [0.5, 4, 0.5, 4],
					'icon-rotation-alignment': 'viewport',
					'icon-pitch-alignment': 'viewport'
				},
				paint: {
					'text-color': c.text
				}
			},
			{
				id: 'road-name-text',
				type: 'symbol',
				source: 'osm',
				'source-layer': 'road',
				minzoom: 14,
				layout: {
					'symbol-placement': 'line',
					'text-field': ['get', 'name'],
					'text-font': ['Noto Sans Display Regular'],
					'text-size': 9
				},
				paint: {
					'text-halo-width': 11,
					'text-halo-color': c.textHalo,
					'text-color': c.text
				}
			},
			{
				id: 'towns',
				type: 'symbol',
				source: 'osm',
				'source-layer': 'cities',
				filter: ['!=', ['get', 'place'], 'city'],
				layout: {
					// "symbol-sort-key": ["get", "population"],
					'text-field': ['get', 'name'],
					'text-font': ['Noto Sans Display Regular'],
					'text-size': 12
				},
				paint: {
					'text-halo-width': 1,
					'text-halo-color': c.textHalo,
					'text-color': c.text
				}
			},
			{
				id: 'cities',
				type: 'symbol',
				source: 'osm',
				'source-layer': 'cities',
				filter: ['==', ['get', 'place'], 'city'],
				layout: {
					'symbol-sort-key': ['-', ['coalesce', ['get', 'population'], 0]],
					'text-field': ['get', 'name'],
					'text-font': ['Noto Sans Display Bold'],
					'text-size': ['interpolate', ['linear'], ['zoom'], 6, 12, 9, 16]
				},
				paint: {
					'text-halo-width': 2,
					'text-halo-color': 'white',
					'text-color': 'hsl(0, 0%, 20%)'
				}
				// }, {
				//    "id": "tiles_debug_info_bound",
				//    "type": "line",
				//    "source": "osm",
				//    "source-layer": "tiles_debug_info",
				//    "paint": {
				//      "line-color": "magenta",
				//      "line-width": 1
				//    }
				// }, {
				//     "id": "tiles_debug_info_name",
				//     "type": "symbol",
				//     "source": "osm",
				//     "source-layer": "tiles_debug_info",
				//     "layout": {
				//       "text-field": ["get", "tile_id"],
				//       "text-font": ["Noto Sans Display Bold"],
				//       "text-size": 16,
				//     },
				//     "paint": {
				//       "text-halo-width": 2,
				//       "text-halo-color": "white",
				//       "text-color": "hsl(0, 0%, 20%)"
				//     }
			}
		]
	};
};
